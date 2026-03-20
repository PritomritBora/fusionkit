# Sensor Fusion 3D — Project Progress

## Dataset
- KITTI Raw Drive: `2011_09_26_drive_0001_sync`
- 108 frames, ~11 seconds, ~104m of driving
- 1392x512 images, ~121k LiDAR points per frame
- Location: `data/kitti/raw/2011_09_26/2011_09_26/`
- Note: KITTI zip extracts with a double-nested date folder (`2011_09_26/2011_09_26/`), handled automatically in `KITTIRawLoader`

## Environment
- Conda environment: `sensor-fusion`
- Python 3.11 (required — open3d has no wheel for 3.12/3.13)
- Config: `sensor-fusion-3d/environment.yml`

---

## Stage 1 — Data Ingestion (`src/preprocessing/ingest.py`)
**Status: Done**

### Changes
- Added `KITTIRawLoader` alongside the existing `KITTILoader` (which expects the odometry dataset layout)
- `KITTIRawLoader` handles the raw drive layout: `image_02/data/`, `velodyne_points/data/`, `oxts/data/`
- `load_calib()` parses `calib_cam_to_cam.txt`, `calib_velo_to_cam.txt`, `calib_imu_to_velo.txt`
  - Returns `P2`, `R0_rect`, `Tr_velo_to_cam`, `imu_to_velo`
  - Skips non-numeric header lines (e.g. `calib_date: 09-Jan-2012`)
- `load_oxts()` returns GPS/IMU data as a dict (lat, lon, alt, roll, pitch, yaw, accelerations, etc.)

### Run
```bash
python sensor-fusion-3d/src/preprocessing/ingest.py \
  --mode raw --data_path data/kitti/raw \
  --date 2011_09_26 --drive 2011_09_26_drive_0001
```

---

## Stage 2 — Sensor Calibration (`src/calibration/calibrate.py`)
**Status: Done**

### What it does
- Projects LiDAR points into camera image space using `Tr_velo_to_cam`, `R0_rect`, `P2`
- Colors projected points with RGB from the camera image
- Two visualization modes:
  - `--mode rgb`: only points inside camera FOV, colored from image (~10-15% of scan)
  - `--mode depth`: all 360° LiDAR points colored by distance (blue=near, red=far)

### Changes
- Updated `__main__` to use `KITTIRawLoader` instead of `KITTILoader`
- Added `sys.path` fix for running as a plain script
- Added `depth_color_point_cloud()` for full-scan visualization

### Run
```bash
python sensor-fusion-3d/src/calibration/calibrate.py \
  --data_path data/kitti/raw --date 2011_09_26 \
  --drive 2011_09_26_drive_0001 --frame 0 --mode depth
```

---

## Stage 3 — Visual Odometry (`src/localization/odometry.py`)
**Status: Done**

### What it does
- ORB feature matching between consecutive frames
- Essential matrix decomposition via RANSAC to estimate R, t
- Accumulates camera-to-world poses across all 108 frames
- Saves trajectory in KITTI format (N x 12)

### Changes
- Updated `__main__` to use `KITTIRawLoader`
- Fixed pose composition bug: was `R_new = R @ R_prev`, corrected to `R_new = R_prev @ R.T` with `t_new = t_prev - R_new @ t` for proper camera-to-world accumulation

### Run
```bash
python sensor-fusion-3d/src/localization/odometry.py \
  --data_path data/kitti/raw --date 2011_09_26 \
  --drive 2011_09_26_drive_0001
```
Output: `results/trajectory_2011_09_26_drive_0001.txt`

---

## Stage 4 — EKF Fusion (`src/fusion/ekf.py`, `src/fusion/run_fusion.py`)
**Status: Done**

### What it does
- Fuses VO position estimates with GPS/IMU from oxts
- EKF state: `[x, y, z, vx, vy, vz]`
- Predict step: IMU forward/left/up acceleration
- Update step: VO position measurement
- Plots three trajectories for comparison: VO only, GPS only, EKF fused

### Result
- EKF trajectory sits between VO (smooth but drifting) and GPS (noisy but globally consistent) — correct expected behavior

### Run
```bash
python sensor-fusion-3d/src/fusion/run_fusion.py \
  --data_path data/kitti/raw --date 2011_09_26 \
  --drive 2011_09_26_drive_0001
```
Output: `results/fusion_trajectory.png`, `results/trajectory_2011_09_26_drive_0001.txt`

---

## Stage 5 — 3D Map Reconstruction (`src/reconstruction/build_map.py`)
**Status: Done (with known limitations)**

### What it does
- Accumulates all 108 LiDAR frames into a single global point cloud
- Per-frame pose from GPS position + IMU roll/pitch/yaw (world-frame consistent)
- Transform chain: LiDAR → IMU (via `imu_to_velo` inverse) → world (via GPS+IMU pose)
- Points in camera FOV get RGB colors from image; out-of-FOV points get grey
- Statistical outlier removal to clean up isolated noise points
- Voxel downsampling at 0.2m for memory management

### Result
- Clean ~100m road corridor visible in bird's eye view
- RGB-colored structures on forward-facing surfaces
- Bird's eye view saved to `results/map_birdseye.png`

### Run
```bash
python sensor-fusion-3d/src/reconstruction/build_map.py \
  --data_path data/kitti/raw --date 2011_09_26 \
  --drive 2011_09_26_drive_0001
```
Output: `results/map_2011_09_26_drive_0001.pcd`, `results/map_birdseye.png`

### Known limitations
- Only 108 frames (~11s) gives limited scene coverage
- LiDAR only captures visible surfaces — backsides of buildings are missing
- Uneven point density (dense near car, sparse far away)

---

## Pending Stages

### nuScenes Support (Phase 2)
**Status: Ready to test — needs data**

`NuScenesLoader` is fully implemented in `src/io/nuscenes_loader.py`:
- CAM_FRONT images (1600x900)
- LIDAR_TOP point clouds (N, 5) → (N, 4) dropping ring channel
- Calibration from sensor metadata (intrinsics + LiDAR-to-camera extrinsic)
- `load_pose_hint()` from ego_pose — XY metres encoded as synthetic lat/lon for EKF compatibility
- `run.py` fully wired: `build_loader()` + `run_with_loader()` handle nuScenes generically

**To run:**
```bash
# 1. Download nuScenes mini (~4 GB, free):
#    https://www.nuscenes.org/nuscenes#download
#    Extract to: data/nuscenes/

# 2. Install devkit (already in requirements.txt):
pip install nuscenes-devkit pyquaternion

# 3. Run:
python sensor-fusion-3d/run.py --config sensor-fusion-3d/configs/nuscenes.yaml
```

**Known nuScenes limitations:**
- Keyframes only (~2 Hz) — VO will have fewer feature matches than KITTI's 10 Hz
- No raw IMU per keyframe — EKF predict step uses zero acceleration (ego_pose only)
- Synthetic lat/lon encoding means GPS reference is in metres, not degrees
- Poisson surface reconstruction from point cloud
- Export as `.obj` for Unreal Engine
- **Recommendation**: skip mesh, export as `.las`/`.laz` instead and use Unreal's Lidar Point Cloud Plugin for better visual quality given the sparse/uneven point cloud

### Evaluation (`src/evaluation/evaluate.py`)
- Compare estimated trajectory against GPS ground truth
- Compute ATE (Absolute Trajectory Error) and RTE (Relative Trajectory Error)
- Not yet run

---

## Results Summary

| File | Description |
|------|-------------|
| `results/trajectory_2011_09_26_drive_0001.txt` | VO trajectory (108 poses) |
| `results/fusion_trajectory.png` | VO vs GPS vs EKF comparison plot |
| `results/map_2011_09_26_drive_0001.pcd` | Global point cloud map |
| `results/map_birdseye.png` | Bird's eye view of the map |
| `results/images/pcd_depth_frame000000.pcd` | Single-frame depth-colored point cloud |
