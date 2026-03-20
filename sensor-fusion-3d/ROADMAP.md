# FusionKit — Roadmap

## Vision

A modular, dataset-agnostic sensor fusion and 3D reconstruction toolkit.
Switch datasets by changing a config file. Same pipeline, same code.

```
python run.py --config configs/kitti.yaml
python run.py --config configs/nuscenes.yaml
python run.py --config configs/my_rosbag.yaml
```

---

## Architecture

```
configs/
  kitti.yaml
  nuscenes.yaml
  rosbag.yaml

src/
  io/
    base_loader.py        ← abstract interface (all datasets implement this)
    kitti_loader.py       ← KITTI raw + odometry
    nuscenes_loader.py    ← nuScenes
    euroc_loader.py       ← EuRoC MAV
    rosbag_loader.py      ← any ROS bag

  calibration/
    calibrate.py          ← LiDAR-camera projection, colored point clouds

  localization/
    odometry.py           ← visual odometry (ORB + essential matrix)

  fusion/
    ekf.py                ← Extended Kalman Filter
    run_fusion.py         ← fuses VO + GPS + IMU

  reconstruction/
    build_map.py          ← accumulates LiDAR frames into global map
    mesh_export.py        ← Poisson mesh, .obj export
    gaussian_splats.py    ← 3D Gaussian splatting (Phase 4)

  evaluation/
    evaluate.py           ← ATE, RTE, drift metrics + plots

run.py                    ← single entry point, config-driven
```

---

## Phase 1 — Config-driven pipeline (current focus)
> Goal: "same code, different configs"

- [x] Wire `KITTIRawLoader` to inherit `BaseLoader`
- [x] Create `run.py` — single entry point that reads a YAML config and runs the full pipeline
- [x] Create `configs/kitti.yaml` — KITTI raw drive config
- [x] Config controls: which sensors, fusion method, reconstruction type, output options
- [x] Update README with `run.py` usage

**Done when:** `python run.py --config configs/kitti.yaml` runs the full pipeline end-to-end.

---

## Phase 2 — nuScenes support
> Goal: second dataset working with zero pipeline changes

- [x] Implement `NuScenesLoader(BaseLoader)`
  - images from 6 cameras (use front camera)
  - LiDAR from `LIDAR_TOP`
  - IMU + GPS from CAN bus data
  - calibration from sensor metadata
- [x] Create `configs/nuscenes.yaml`
- [ ] Test on a mini scene (nuScenes mini is ~4GB, free download)
- [ ] Verify same VO + EKF + map pipeline produces valid output

**Done when:** `python run.py --config configs/nuscenes.yaml` works.

---

## Phase 3 — ROS bag support
> Goal: works on any robot/custom hardware

- [ ] Implement `ROSBagLoader(BaseLoader)`
  - configurable topic names (camera, lidar, imu, gps)
  - handles message synchronization across topics
  - supports rospy or rosbags library (no ROS install required)
- [ ] Create `configs/rosbag_template.yaml` with topic name placeholders
- [ ] Test on a public ROS bag (e.g. TUM RGB-D, or any Velodyne bag)

**Done when:** any ROS bag with camera + LiDAR topics can be processed.

---

## Phase 4 — Advanced outputs
> Goal: visual wow factor + Unreal Engine integration

- [ ] Gaussian splatting output (`src/reconstruction/gaussian_splats.py`)
  - convert point cloud to 3D Gaussian splats
  - export for real-time rendering
- [ ] Unreal Engine export
  - `.las`/`.laz` point cloud export for Lidar Point Cloud Plugin
  - `.obj` mesh export from Poisson reconstruction
- [ ] Real-time Open3D visualization during pipeline run

---

## Phase 5 — Polish & portfolio
> Goal: production-quality repo

- [ ] Fix EKF coordinate frame mismatch (transform VO to ENU before update)
- [ ] Add ICP refinement between consecutive LiDAR frames in map builder
- [ ] Loop closure detection (basic bag-of-words)
- [ ] CI with GitHub Actions (lint + unit tests)
- [ ] Screen recording of 3D map rotating (embed in README as GIF)
- [ ] Write Medium article / project writeup

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| KITTI Raw loader | done | handles double-nested zip layout |
| Sensor calibration | done | LiDAR→camera projection, depth coloring |
| Visual odometry | done | ORB + essential matrix, 108 frames |
| EKF fusion | done | VO + GPS + IMU, trajectory plot |
| 3D map reconstruction | done | GPS+IMU poses, RGB-colored point cloud |
| Evaluation (ATE/RTE) | done | VO: 3.6m ATE, EKF: 28m ATE (frame mismatch known issue) |
| BaseLoader interface | done | abstract base, all loaders inherit |
| Config-driven run.py | done | `python run.py --config configs/kitti.yaml` works |
| nuScenes loader | done | NuScenesLoader implemented, wired to run.py |
| nuScenes config | done | configs/nuscenes.yaml |
| nuScenes tested | pending | needs data download |
| ROS bag loader | not started | Phase 3 |
| Gaussian splats | not started | Phase 4 |

---

## Key Design Decisions

**Why EKF over particle filter?**
EKF is O(n) vs O(n*particles). For 6-DOF state with GPS+IMU+VO, EKF is fast enough and well-understood. Particle filters shine for highly nonlinear/multimodal distributions (e.g. global localization from scratch).

**Why monocular VO over stereo?**
KITTI raw provides stereo but monocular VO is harder and more general — most robots have a single camera. The scale ambiguity is handled by GPS fusion in the EKF.

**Why GPS as ground truth for evaluation?**
KITTI raw doesn't include ground truth poses (that's the odometry dataset). GPS with Umeyama alignment is a reasonable proxy for short drives. For production evaluation, RTK GPS or LiDAR-based SLAM would be used.

**Why point cloud over mesh for the map?**
Poisson mesh reconstruction requires dense, uniformly sampled points with consistent normals. LiDAR scans are sparse and uneven — the mesh artifacts would be worse than the point cloud. Gaussian splats are the better path for visual quality.
