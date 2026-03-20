"""
Stage 4 — EKF Fusion Runner
Fuses VO trajectory with GPS/IMU from oxts.
Plots VO-only, GPS-only, and EKF-fused trajectories for comparison.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.fusion.ekf import EKF
from src.localization.odometry import VisualOdometry, extract_intrinsics, save_trajectory
from src.preprocessing.ingest import KITTIRawLoader


def latlon_to_xy(lat, lon, lat0, lon0):
    """Approximate GPS lat/lon to local XY in metres."""
    R = 6371000.0
    x = np.radians(lon - lon0) * R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R
    return x, y


def run(data_path, date, drive, max_frames):
    loader = KITTIRawLoader(data_path, date, drive)
    calib  = loader.load_calib()
    K      = extract_intrinsics(calib)
    n      = min(max_frames, len(loader))

    vo  = VisualOdometry(K)
    ekf = EKF(dt=0.1)

    vo_traj, gps_traj, ekf_traj = [], [], []

    # Anchor GPS origin at frame 0
    oxts0  = loader.load_oxts(0)
    lat0, lon0 = oxts0["lat"], oxts0["lon"]

    for i in range(n):
        frame = loader.load_image(i)
        oxts  = loader.load_oxts(i)

        # --- VO ---
        pose = vo.process_frame(frame)
        vo_pos = pose[:, 3]          # translation column (x,y,z in camera frame)
        vo_traj.append(vo_pos.copy())

        # --- GPS (local XY, keep Z from altitude) ---
        gx, gy = latlon_to_xy(oxts["lat"], oxts["lon"], lat0, lon0)
        gz = oxts["alt"] - oxts0["alt"]
        gps_traj.append(np.array([gx, gy, gz]))

        # --- EKF: predict with IMU accel, update with VO ---
        accel = np.array([oxts["af"], oxts["al"], oxts["au"]])  # forward/left/up
        ekf.predict(accel)
        ekf.update_vo(vo_pos)
        ekf_traj.append(ekf.position.copy())

    vo_traj  = np.array(vo_traj)
    gps_traj = np.array(gps_traj)
    ekf_traj = np.array(ekf_traj)

    return vo_traj, gps_traj, ekf_traj, vo.poses


def plot(vo_traj, gps_traj, ekf_traj):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Top-down XZ (camera coords: X=right, Z=forward)
    ax = axes[0]
    ax.plot(vo_traj[:, 0],  vo_traj[:, 2],  label="VO only",   color="steelblue")
    ax.plot(gps_traj[:, 0], gps_traj[:, 1], label="GPS only",  color="tomato",    linestyle="--")
    ax.plot(ekf_traj[:, 0], ekf_traj[:, 2], label="EKF fused", color="seagreen",  linewidth=2)
    ax.scatter([0], [0], color="black", zorder=5, label="Start")
    ax.set_title("Top-down trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z / forward (m)")
    ax.legend()
    ax.grid(True)
    ax.set_aspect("equal")

    # Altitude / height over frames
    ax = axes[1]
    frames = np.arange(len(vo_traj))
    ax.plot(frames, vo_traj[:, 1],  label="VO Y",      color="steelblue")
    ax.plot(frames, gps_traj[:, 2], label="GPS alt",   color="tomato",   linestyle="--")
    ax.plot(frames, ekf_traj[:, 1], label="EKF Y",     color="seagreen", linewidth=2)
    ax.set_title("Height over frames")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Height (m)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    Path("results").mkdir(exist_ok=True)
    plt.savefig("results/fusion_trajectory.png", dpi=150)
    print("Saved: results/fusion_trajectory.png")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  default="data/kitti/raw")
    parser.add_argument("--date",       default="2011_09_26")
    parser.add_argument("--drive",      default="2011_09_26_drive_0001")
    parser.add_argument("--max_frames", type=int, default=108)
    args = parser.parse_args()

    vo_traj, gps_traj, ekf_traj, poses = run(
        args.data_path, args.date, args.drive, args.max_frames
    )

    save_trajectory(poses, f"results/trajectory_{args.drive}.txt")
    plot(vo_traj, gps_traj, ekf_traj)
