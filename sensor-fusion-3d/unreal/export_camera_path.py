"""
Stage 6 — Unreal Engine Camera Path Export
Convert estimated trajectory poses to a CSV format compatible with
Unreal Engine Sequencer (Level Sequence camera keyframes).

Format: frame, x, y, z, pitch, yaw, roll (in Unreal coordinate space)
KITTI uses right-hand Z-forward; Unreal uses left-hand Z-up.
"""

import numpy as np
import csv
from pathlib import Path
from scipy.spatial.transform import Rotation


KITTI_TO_UNREAL_SCALE = 100.0  # meters to centimeters


def rotation_to_euler_unreal(R: np.ndarray) -> tuple:
    """Convert 3x3 rotation matrix to Unreal pitch/yaw/roll (degrees)."""
    # KITTI: X-right, Y-down, Z-forward
    # Unreal: X-forward, Y-right, Z-up
    # Remap axes
    R_ue = np.array([
        [ R[2, 2],  R[2, 0], -R[2, 1]],
        [ R[0, 2],  R[0, 0], -R[0, 1]],
        [-R[1, 2], -R[1, 0],  R[1, 1]],
    ])
    rot = Rotation.from_matrix(R_ue)
    pitch, yaw, roll = rot.as_euler("YXZ", degrees=True)
    return pitch, yaw, roll


def kitti_pos_to_unreal(t: np.ndarray) -> tuple:
    """Remap KITTI position (x, y, z) to Unreal (x, y, z) in cm."""
    x_ue = t[2] * KITTI_TO_UNREAL_SCALE   # Z-forward -> X
    y_ue = t[0] * KITTI_TO_UNREAL_SCALE   # X-right   -> Y
    z_ue = -t[1] * KITTI_TO_UNREAL_SCALE  # Y-down    -> -Z (up)
    return x_ue, y_ue, z_ue


def export_camera_path(pose_file: str, out_csv: str):
    raw = np.loadtxt(pose_file)
    poses = raw.reshape(-1, 3, 4)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "z", "pitch", "yaw", "roll"])

        for i, pose in enumerate(poses):
            R = pose[:3, :3]
            t = pose[:3, 3]
            x, y, z = kitti_pos_to_unreal(t)
            pitch, yaw, roll = rotation_to_euler_unreal(R)
            writer.writerow([i, f"{x:.4f}", f"{y:.4f}", f"{z:.4f}",
                             f"{pitch:.4f}", f"{yaw:.4f}", f"{roll:.4f}"])

    print(f"Camera path exported: {out_csv} ({len(poses)} keyframes)")
    print("Import into Unreal via: Sequencer > Camera > Import CSV Keyframes")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", default="results/trajectory_seq00.txt")
    parser.add_argument("--out", default="unreal/camera_path.csv")
    args = parser.parse_args()

    export_camera_path(args.poses, args.out)
