# Multi-Sensor 3D Scene Reconstruction and Cinematic Visualization

Fuse camera (RGB), LiDAR (depth), and IMU (motion) data to reconstruct a precise 3D environment and camera trajectory, then visualize it inside Unreal Engine.

## Pipeline Overview

```
Data Ingestion → Calibration → Localization → Reconstruction → Enhancement → Unreal Engine
```

## Stages

1. **Data Ingestion** — Load and sync RGB, LiDAR, IMU from KITTI/nuScenes
2. **Calibration** — Align sensors into a common coordinate frame
3. **Localization** — Visual odometry + EKF for trajectory estimation
4. **Reconstruction** — Accumulate LiDAR frames into a global point cloud
5. **Enhancement** — Convert to Gaussian Splat or mesh for Unreal
6. **Unreal Integration** — Import mesh + trajectory, cinematic sequencer
7. **Evaluation** — RMSE, drift, density metrics

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Download KITTI odometry dataset: https://www.cvlibs.net/datasets/kitti/eval_odometry.php

Place files under `data/kitti/`.

## Usage

```bash
# Stage 1: Ingest data
python src/preprocessing/ingest.py --dataset kitti --sequence 00

# Stage 2: Calibration
python src/calibration/calibrate.py --sequence 00

# Stage 3: Localization
python src/localization/odometry.py --sequence 00

# Stage 4: Reconstruction
python src/reconstruction/build_map.py --sequence 00

# Evaluate
python src/evaluation/evaluate.py --sequence 00
```

## Tech Stack

- Python, NumPy, OpenCV, Open3D
- Optional: PyTorch, ROS
- Unreal Engine 5 (Stage 6)
