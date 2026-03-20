"""
NuScenesLoader — loads nuScenes mini/full dataset via the nuscenes-devkit.

Implements BaseLoader so the entire pipeline (calibration, VO, EKF, mapping)
runs unchanged. Uses CAM_FRONT + LIDAR_TOP, ego_pose for GPS/IMU hints.

Install:
    pip install nuscenes-devkit

Download nuScenes mini (~4 GB, free):
    https://www.nuscenes.org/nuscenes#download
    Extract to:  data/nuscenes/

Usage:
    loader = NuScenesLoader("data/nuscenes", version="v1.0-mini", scene_index=0)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from pyquaternion import Quaternion

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.io.base_loader import BaseLoader


class NuScenesLoader(BaseLoader):
    """
    Loads a single nuScenes scene as a frame sequence.

    Each "frame" is one keyframe sample in the scene.
    Camera  : CAM_FRONT  (1600x900)
    LiDAR   : LIDAR_TOP  (N, 4) — x, y, z, intensity
    Pose    : ego_pose   — translated to BaseLoader pose_hint format
    """

    CAMERA  = "CAM_FRONT"
    LIDAR   = "LIDAR_TOP"

    def __init__(self, dataroot: str, version: str = "v1.0-mini", scene_index: int = 0):
        from nuscenes.nuscenes import NuScenes
        self.nusc     = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.dataroot = Path(dataroot)

        scene         = self.nusc.scene[scene_index]
        self._samples = self._collect_samples(scene)

        # Cache calibration (same for all frames in a scene)
        self._calib: dict | None = None

        # Reference ego pose for local-XY origin (frame 0)
        self._ego0: dict | None = None

    # ── internal helpers ────────────────────────────────────────────────────

    def _collect_samples(self, scene: dict) -> list[dict]:
        """Walk the linked-list of samples and return them in order."""
        samples = []
        token = scene["first_sample_token"]
        while token:
            sample = self.nusc.get("sample", token)
            samples.append(sample)
            token = sample["next"]
        return samples

    def _sample_data(self, idx: int, channel: str) -> dict:
        return self.nusc.get("sample_data", self._samples[idx]["data"][channel])

    def _ego_pose(self, idx: int, channel: str) -> dict:
        sd   = self._sample_data(idx, channel)
        return self.nusc.get("ego_pose", sd["ego_pose_token"])

    def _sensor_calib(self, channel: str) -> dict:
        sd = self._sample_data(0, channel)
        return self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

    # ── BaseLoader interface ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def load_image(self, idx: int) -> np.ndarray:
        sd   = self._sample_data(idx, self.CAMERA)
        path = self.dataroot / sd["filename"]
        img  = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return img

    def load_lidar(self, idx: int) -> np.ndarray:
        """Returns (N, 4) float32: x, y, z, intensity — same layout as KITTI."""
        sd   = self._sample_data(idx, self.LIDAR)
        path = self.dataroot / sd["filename"]
        # nuScenes LiDAR binary: 5 floats per point (x,y,z,intensity,ring)
        raw  = np.fromfile(str(path), dtype=np.float32).reshape(-1, 5)
        return raw[:, :4]   # drop ring channel

    def load_calib(self) -> dict:
        """
        Build a KITTI-compatible calib dict from nuScenes sensor metadata.

        Keys returned:
            P2              (3, 4)  camera projection matrix
            R0_rect         (3, 3)  identity (nuScenes images are already rectified)
            Tr_velo_to_cam  (3, 4)  LiDAR-to-camera extrinsic
            imu_to_velo     (3, 4)  ego-to-LiDAR (approximation — nuScenes has no IMU extrinsic)
        """
        if self._calib is not None:
            return self._calib

        cam_calib  = self._sensor_calib(self.CAMERA)
        lidar_calib = self._sensor_calib(self.LIDAR)

        # ── Camera intrinsics → P2 ──────────────────────────────────────────
        K = np.array(cam_calib["camera_intrinsic"], dtype=np.float64)  # (3, 3)
        P2 = np.hstack([K, np.zeros((3, 1))])                          # (3, 4)

        # ── LiDAR-to-camera extrinsic ───────────────────────────────────────
        # nuScenes stores each sensor's pose in ego frame.
        # Tr_velo_to_cam = T_cam_from_ego  @  T_ego_from_lidar
        #                = inv(T_ego_from_cam) @ T_ego_from_lidar
        R_lidar = Quaternion(lidar_calib["rotation"]).rotation_matrix
        t_lidar = np.array(lidar_calib["translation"])

        R_cam   = Quaternion(cam_calib["rotation"]).rotation_matrix
        t_cam   = np.array(cam_calib["translation"])

        # T_ego_from_lidar  (4x4)
        T_ego_lidar = _make_T(R_lidar, t_lidar)
        # T_ego_from_cam    (4x4)
        T_ego_cam   = _make_T(R_cam, t_cam)
        # T_cam_from_lidar  (4x4)
        T_cam_lidar = np.linalg.inv(T_ego_cam) @ T_ego_lidar

        Tr_velo_to_cam = T_cam_lidar[:3, :]   # (3, 4)

        # ── imu_to_velo: identity approximation ────────────────────────────
        # nuScenes ego frame ≈ IMU frame; LiDAR is mounted with known offset.
        # Use inv(T_ego_from_lidar) as a reasonable stand-in.
        imu_to_velo = np.linalg.inv(T_ego_lidar)[:3, :]   # (3, 4)

        self._calib = {
            "P2":             P2,
            "R0_rect":        np.eye(3),
            "Tr_velo_to_cam": Tr_velo_to_cam,
            "imu_to_velo":    imu_to_velo,
        }
        return self._calib

    def load_pose_hint(self, idx: int) -> dict:
        """
        Return a BaseLoader-compatible pose hint from ego_pose.

        nuScenes ego_pose gives translation + quaternion in global frame.
        We convert to the same keys the EKF / map builder expect:
            lat, lon, alt   — synthesized from XY metres relative to frame 0
            roll, pitch, yaw
            ax, ay, az, af, al, au  — zeros (no raw IMU in nuScenes keyframes)

        Encoding: lat = dy/R * (180/π),  lon = dx/R * (180/π)
        where dx, dy are metres relative to frame 0.
        latlon_to_xy(lat, lon, lat0=0, lon0=0) then recovers dx, dy exactly
        because cos(lat0=0) = 1.
        """
        ego = self._ego_pose(idx, self.CAMERA)

        # Anchor origin at frame 0
        if self._ego0 is None:
            self._ego0 = self._ego_pose(0, self.CAMERA)

        tx0, ty0, tz0 = self._ego0["translation"]
        tx,  ty,  tz  = ego["translation"]

        # Relative displacement in metres
        dx = tx - tx0
        dy = ty - ty0
        dz = tz - tz0

        # Quaternion → Euler (roll, pitch, yaw)
        q = Quaternion(ego["rotation"])
        roll, pitch, yaw = _quat_to_rpy(q)

        # Encode relative XY as fake lat/lon anchored at 0,0
        # latlon_to_xy(lat, lon, 0, 0) = (lon*R*π/180, lat*R*π/180) = (dx, dy) ✓
        R_earth = 6_371_000.0
        lat = dy / R_earth * (180.0 / np.pi)
        lon = dx / R_earth * (180.0 / np.pi)

        return {
            "lat":   lat,
            "lon":   lon,
            "alt":   dz,          # relative altitude
            "roll":  roll,
            "pitch": pitch,
            "yaw":   yaw,
            # IMU accelerations — not available per-keyframe in nuScenes
            "ax": 0.0, "ay": 0.0, "az": 0.0,
            "af": 0.0, "al": 0.0, "au": 0.0,
            # Velocity — not available per-keyframe
            "vn": 0.0, "ve": 0.0, "vf": 0.0, "vl": 0.0, "vu": 0.0,
        }

    def load_timestamps(self) -> np.ndarray:
        """Returns per-frame timestamps in seconds."""
        ts = []
        for sample in self._samples:
            sd = self.nusc.get("sample_data", sample["data"][self.CAMERA])
            ts.append(sd["timestamp"] / 1e6)   # microseconds → seconds
        return np.array(ts)

    def load_ego_poses(self) -> list[np.ndarray]:
        """
        Return per-frame 4x4 ego-to-world transforms directly from ego_pose.
        These are exact — no Euler angle conversion, no lat/lon encoding.
        Used by accumulate_map instead of the imu_pose() approximation.
        """
        poses = []
        ego0 = self._ego_pose(0, self.CAMERA)
        T_world_ego0 = _make_T(
            Quaternion(ego0["rotation"]).rotation_matrix,
            np.array(ego0["translation"])
        )
        T_ego0_world = np.linalg.inv(T_world_ego0)

        for i in range(len(self._samples)):
            ego = self._ego_pose(i, self.CAMERA)
            T_world_ego = _make_T(
                Quaternion(ego["rotation"]).rotation_matrix,
                np.array(ego["translation"])
            )
            # Relative to frame 0 so the map starts at origin
            poses.append(T_ego0_world @ T_world_ego)
        return poses

    # ── convenience ────────────────────────────────────────────────────────

    def scene_name(self) -> str:
        return self.nusc.scene[0]["name"]

    def list_scenes(self) -> list[str]:
        return [s["name"] for s in self.nusc.scene]


# ── module-level helpers ────────────────────────────────────────────────────

def _make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous transform from (3,3) R and (3,) t."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


def _quat_to_rpy(q: "Quaternion") -> tuple[float, float, float]:
    """Convert pyquaternion Quaternion to (roll, pitch, yaw) in radians."""
    R = q.rotation_matrix
    pitch = np.arcsin(-R[2, 0])
    roll  = np.arctan2(R[2, 1], R[2, 2])
    yaw   = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


# ── smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot",     default="data/nuscenes")
    parser.add_argument("--version",      default="v1.0-mini")
    parser.add_argument("--scene_index",  type=int, default=0)
    args = parser.parse_args()

    loader = NuScenesLoader(args.dataroot, args.version, args.scene_index)
    print(f"Scene: {loader.scene_name()}  —  {len(loader)} keyframes")

    img   = loader.load_image(0)
    pts   = loader.load_lidar(0)
    pose  = loader.load_pose_hint(0)
    calib = loader.load_calib()

    print(f"Image  : {img.shape}")
    print(f"LiDAR  : {pts.shape}")
    print(f"Pose   : lat={pose['lat']:.6f}  lon={pose['lon']:.6f}  yaw={pose['yaw']:.3f} rad")
    print(f"P2     :\n{calib['P2']}")
    print(f"Tr_velo_to_cam:\n{calib['Tr_velo_to_cam']}")
