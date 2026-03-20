"""
Stage 7 — Evaluation
Compute trajectory RMSE and drift against ground truth poses.
Compare LiDAR-only vs sensor fusion trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_trajectory(path: str) -> np.ndarray:
    """Load (N, 3, 4) poses from KITTI-format file, return (N, 3) positions."""
    raw = np.loadtxt(path)
    return raw.reshape(-1, 3, 4)[:, :3, 3]


def rmse(estimated: np.ndarray, ground_truth: np.ndarray) -> float:
    """Root mean square error between two (N, 3) position arrays."""
    n = min(len(estimated), len(ground_truth))
    diff = estimated[:n] - ground_truth[:n]
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def final_drift(estimated: np.ndarray, ground_truth: np.ndarray) -> float:
    """Euclidean distance between final estimated and GT positions."""
    n = min(len(estimated), len(ground_truth))
    return float(np.linalg.norm(estimated[n - 1] - ground_truth[n - 1]))


def plot_trajectories(trajectories: dict, out_path: str = "results/images/trajectory.png"):
    """Plot top-down (X-Z) view of trajectories."""
    plt.figure(figsize=(10, 8))
    for label, traj in trajectories.items():
        plt.plot(traj[:, 0], traj[:, 2], label=label)
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.title("Trajectory Comparison (Top-Down View)")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Trajectory plot saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    import argparse
    from src.preprocessing.ingest import KITTILoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--data_path", default="data/kitti")
    parser.add_argument("--estimated", default=None,
                        help="Path to estimated trajectory file")
    args = parser.parse_args()

    loader = KITTILoader(args.data_path, args.sequence)
    gt_poses = loader.load_poses()
    gt_traj = gt_poses[:, :3, 3]

    trajectories = {"Ground Truth": gt_traj}

    if args.estimated:
        est_traj = load_trajectory(args.estimated)
        trajectories["Estimated"] = est_traj
        print(f"RMSE:        {rmse(est_traj, gt_traj):.4f} m")
        print(f"Final Drift: {final_drift(est_traj, gt_traj):.4f} m")
    else:
        # Default: load from standard output path
        est_path = f"results/trajectory_seq{args.sequence}.txt"
        if Path(est_path).exists():
            est_traj = load_trajectory(est_path)
            trajectories["VO Estimated"] = est_traj
            print(f"RMSE:        {rmse(est_traj, gt_traj):.4f} m")
            print(f"Final Drift: {final_drift(est_traj, gt_traj):.4f} m")

    plot_trajectories(trajectories)
