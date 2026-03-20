# FusionKit

A modular, dataset-agnostic sensor fusion pipeline for 3D mapping and localization.
Combines camera, LiDAR, GPS, and IMU data to produce fused trajectories and 3D point cloud maps.

## The Problem

Most sensor fusion implementations are tightly coupled to a single dataset format.
Switching from KITTI to nuScenes, or from a recorded dataset to a live ROS bag, requires
rewriting large parts of the pipeline. FusionKit solves this by separating the data layer
from the algorithms — plug in any data source, get the same pipeline.

## What it does

```
Camera ──────┐
LiDAR  ──────┼──► Calibration ──► Visual Odometry ──► EKF Fusion ──► 3D Map
GPS/IMU ─────┘                                    ↗
```

- calibration: projects LiDAR points into camera frame, produces colored point clouds
- visual odometry: ORB feature matching + essential matrix decomposition for pose estimation
- EKF fusion: fuses VO position with GPS/IMU to correct drift
- 3D mapping: accumulates RGB-colored LiDAR frames into a global point cloud map

## Current Status

| Stage | Status |
|-------|--------|
| Data ingestion (KITTI Raw) | done |
| Sensor calibration | done |
| Visual odometry | done |
| EKF fusion (VO + GPS + IMU) | done |
| 3D map reconstruction | done |
| Evaluation (ATE/RTE) | pending |
| Mesh export | pending |
| nuScenes loader | planned |
| EuRoC loader | planned |
| ROS bag loader | planned |

## Planned: Dataset Interface

The goal is a single `BaseLoader` interface that any dataset or live sensor stream implements.
The entire pipeline runs unchanged regardless of the data source.

```python
class BaseLoader(ABC):
    def __len__(self) -> int: ...
    def load_image(self, idx: int) -> np.ndarray: ...       # (H, W, 3)
    def load_lidar(self, idx: int) -> np.ndarray: ...       # (N, 4) x,y,z,intensity
    def load_calib(self) -> dict: ...                       # P2, R0_rect, Tr_velo_to_cam
    def load_pose_hint(self, idx: int) -> dict | None: ...  # GPS/IMU, optional
```

Planned loaders:

- `KITTIRawLoader` — KITTI raw drives (done)
- `KITTIOdometryLoader` — KITTI odometry sequences (done)
- `NuScenesLoader` — nuScenes dataset
- `EuRoCLoader` — EuRoC MAV dataset (camera + IMU, no LiDAR)
- `ROSBagLoader` — any ROS bag with standard sensor topics

## Planned: Config-driven pipeline

```yaml
dataset:
  type: kitti_raw
  data_path: data/kitti/raw
  date: "2011_09_26"
  drive: "2011_09_26_drive_0001"

pipeline:
  voxel_size: 0.2
  orb_features: 2000
  ekf_dt: 0.1

output:
  trajectory: results/trajectory.txt
  map: results/map.pcd
```

## Quickstart (KITTI Raw)

```bash
conda env create -f environment.yml
conda activate sensor-fusion

# Download KITTI raw drive
bash scripts/download_kitti_drive.sh

# Run full pipeline
python sensor-fusion-3d/src/reconstruction/build_map.py \
  --data_path data/kitti/raw \
  --date 2011_09_26 \
  --drive 2011_09_26_drive_0001
```

## Tech stack

- Python 3.11
- OpenCV — feature detection, essential matrix, pose recovery
- Open3D — point cloud processing and visualization
- NumPy / SciPy — linear algebra, EKF
- Matplotlib — trajectory plots
