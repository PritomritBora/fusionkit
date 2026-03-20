"""
BaseLoader — abstract interface for sensor data sources.

Any dataset or live sensor stream implements this interface.
The entire pipeline (calibration, VO, EKF, mapping) operates
on BaseLoader instances without knowing the underlying format.

Planned implementations:
    KITTIRawLoader      — KITTI raw drives (src/preprocessing/ingest.py)
    KITTIOdometryLoader — KITTI odometry sequences
    NuScenesLoader      — nuScenes dataset
    EuRoCLoader         — EuRoC MAV (camera + IMU, no LiDAR)
    ROSBagLoader        — any ROS bag with standard sensor topics
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseLoader(ABC):
    """
    Abstract base class for all sensor data loaders.

    Subclasses must implement the core methods. Pose hints (GPS/IMU)
    are optional — return None if the dataset doesn't provide them.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Total number of frames in the sequence."""
        ...

    @abstractmethod
    def load_image(self, idx: int) -> np.ndarray:
        """
        Load RGB image for frame idx.
        Returns: (H, W, 3) uint8 array in BGR order (OpenCV convention).
        """
        ...

    @abstractmethod
    def load_lidar(self, idx: int) -> np.ndarray:
        """
        Load LiDAR point cloud for frame idx.
        Returns: (N, 4) float32 array — columns: x, y, z, intensity.
        """
        ...

    @abstractmethod
    def load_calib(self) -> dict:
        """
        Load sensor calibration parameters.

        Required keys:
            P2              (3, 4)  camera projection matrix
            R0_rect         (3, 3)  rectification matrix
            Tr_velo_to_cam  (3, 4)  LiDAR-to-camera extrinsic

        Optional keys:
            imu_to_velo     (3, 4)  IMU-to-LiDAR extrinsic
        """
        ...

    def load_pose_hint(self, idx: int) -> dict | None:
        """
        Load GPS/IMU data for frame idx. Optional.

        If provided, expected keys (subset is fine):
            lat, lon, alt           GPS position
            roll, pitch, yaw        IMU orientation (radians)
            ax, ay, az              linear acceleration (m/s^2)
            vx, vy, vz              velocity (m/s)

        Returns None if the dataset has no pose hints.
        """
        return None

    def load_timestamps(self) -> np.ndarray | None:
        """
        Load per-frame timestamps in seconds. Optional.
        Returns (N,) float64 array or None.
        """
        return None
