"""
Stage 1 — Data Ingestion
Load and synchronize RGB images, LiDAR point clouds, and IMU/GPS data.
"""

import numpy as np
import cv2
from pathlib import Path
from src.io.base_loader import BaseLoader


class KITTILoader(BaseLoader):
    """Loads KITTI odometry dataset sequences."""

    def __init__(self, base_path: str, sequence: str):
        self.base = Path(base_path)
        self.seq = sequence.zfill(2)
        self.seq_path = self.base / "sequences" / self.seq
        self.poses_path = self.base / "poses" / f"{self.seq}.txt"

        self.image_dir = self.seq_path / "image_2"
        self.lidar_dir = self.seq_path / "velodyne"
        self.calib_file = self.seq_path / "calib.txt"
        self.times_file = self.seq_path / "times.txt"

    def load_timestamps(self) -> np.ndarray:
        return np.loadtxt(self.times_file)

    def load_image(self, idx: int) -> np.ndarray:
        path = self.image_dir / f"{idx:06d}.png"
        return cv2.imread(str(path))

    def load_lidar(self, idx: int) -> np.ndarray:
        """Returns (N, 4) array: x, y, z, intensity."""
        path = self.lidar_dir / f"{idx:06d}.bin"
        return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)

    def load_calib(self) -> dict:
        calib = {}
        with open(self.calib_file) as f:
            for line in f:
                key, *vals = line.strip().split()
                calib[key.rstrip(":")] = np.array(vals, dtype=np.float64)
        calib["P2"] = calib["P2"].reshape(3, 4)
        calib["R0_rect"] = calib["R0_rect"].reshape(3, 3)
        calib["Tr_velo_to_cam"] = calib["Tr_velo_to_cam"].reshape(3, 4)
        return calib

    def load_poses(self) -> np.ndarray:
        """Returns (N, 3, 4) ground truth poses."""
        raw = np.loadtxt(self.poses_path)
        return raw.reshape(-1, 3, 4)

    def __len__(self):
        return len(list(self.image_dir.glob("*.png")))


class KITTIRawLoader(BaseLoader):
    """
    Loads KITTI raw dataset drives, e.g. 2011_09_26_drive_0001_sync.
    Expected layout:
        base_path/
          2011_09_26/
            2011_09_26_drive_0001_sync/
              image_02/data/*.png
              velodyne_points/data/*.bin
              oxts/data/*.txt
            calib_cam_to_cam.txt
            calib_imu_to_velo.txt
            calib_velo_to_cam.txt
    """

    def __init__(self, base_path: str, date: str, drive: str):
        # KITTI zips extract with an extra nested date folder, handle both layouts
        candidate = Path(base_path) / date / date
        self.base = candidate if candidate.exists() else Path(base_path) / date
        drive_dir = f"{drive}_sync"
        self.drive_path = self.base / drive_dir

        self.image_dir = self.drive_path / "image_02" / "data"
        self.lidar_dir = self.drive_path / "velodyne_points" / "data"
        self.oxts_dir  = self.drive_path / "oxts" / "data"

        self.calib_cam_to_cam  = self.base / "calib_cam_to_cam.txt"
        self.calib_velo_to_cam = self.base / "calib_velo_to_cam.txt"
        self.calib_imu_to_velo = self.base / "calib_imu_to_velo.txt"

        self._frames = sorted(self.image_dir.glob("*.png"))

    def __len__(self):
        return len(self._frames)

    def load_image(self, idx: int) -> np.ndarray:
        return cv2.imread(str(self._frames[idx]))

    def load_lidar(self, idx: int) -> np.ndarray:
        """Returns (N, 4) array: x, y, z, intensity."""
        path = self.lidar_dir / self._frames[idx].name.replace(".png", ".bin")
        return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)

    def load_pose_hint(self, idx: int) -> dict:
        """Returns GPS/IMU data as a BaseLoader-compatible dict."""
        return self.load_oxts(idx)

    def load_oxts(self, idx: int) -> dict:
        """Returns GPS/IMU data as a dict for the given frame."""
        path = self.oxts_dir / self._frames[idx].name.replace(".png", ".txt")
        vals = np.loadtxt(str(path))
        keys = [
            "lat", "lon", "alt", "roll", "pitch", "yaw",
            "vn", "ve", "vf", "vl", "vu",
            "ax", "ay", "az", "af", "al", "au",
            "wx", "wy", "wz", "wf", "wl", "wu",
            "pos_accuracy", "vel_accuracy",
            "navstat", "numsats", "posmode", "velmode", "orimode"
        ]
        return dict(zip(keys, vals))

    def load_calib(self) -> dict:
        """Parse raw calib files into a standard calib dict."""
        def parse_file(path):
            data = {}
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    key, *vals = line.split()
                    try:
                        data[key.rstrip(":")] = np.array(vals, dtype=np.float64)
                    except ValueError:
                        pass
            return data

        cam  = parse_file(self.calib_cam_to_cam)
        velo = parse_file(self.calib_velo_to_cam)
        imu  = parse_file(self.calib_imu_to_velo)

        calib = {}
        calib["P2"]             = cam["P_rect_02"].reshape(3, 4)
        calib["R0_rect"]        = cam["R_rect_00"].reshape(3, 3)
        calib["Tr_velo_to_cam"] = np.hstack([velo["R"].reshape(3, 3), velo["T"].reshape(3, 1)])
        calib["imu_to_velo"]    = np.hstack([imu["R"].reshape(3, 3),  imu["T"].reshape(3, 1)])
        return calib




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["odometry", "raw"], default="raw")
    parser.add_argument("--data_path", default="data/kitti/raw")
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--date", default="2011_09_26")
    parser.add_argument("--drive", default="2011_09_26_drive_0001")
    args = parser.parse_args()

    if args.mode == "raw":
        loader = KITTIRawLoader(args.data_path, args.date, args.drive)
        print(f"Raw drive {args.drive}: {len(loader)} frames")
        img   = loader.load_image(0)
        pts   = loader.load_lidar(0)
        oxts  = loader.load_pose_hint(0)
        calib = loader.load_calib()
        print(f"Image shape:    {img.shape}")
        print(f"LiDAR points:   {pts.shape[0]}")
        print(f"GPS lat/lon:    {oxts['lat']:.6f}, {oxts['lon']:.6f}")
        print(f"P2:\n{calib['P2']}")
    else:
        loader = KITTILoader(args.data_path, args.sequence)
        print(f"Sequence {args.sequence}: {len(loader)} frames")
        img = loader.load_image(0)
        pts = loader.load_lidar(0)
        print(f"Image shape:  {img.shape}")
        print(f"LiDAR points: {pts.shape[0]}")
