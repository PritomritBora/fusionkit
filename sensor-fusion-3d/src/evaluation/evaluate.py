"""
Stage 6 — Evaluation
Computes standard trajectory metrics against GPS ground truth:
  - ATE (Absolute Trajectory Error): global consistency
  - RTE (Relative Trajectory Error): local drift per 100 frames
  - Final drift: end-point error
Compares VO-only vs EKF-fused trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# --- Metrics ---

def align_trajectories(estimated: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Align estimated trajectory to reference using Umeyama method (no scale).
    Works with 2D or 3D trajectories.
    Returns aligned (N, D) estimated positions.
    """
    n = min(len(estimated), len(reference))
    est, ref = estimated[:n], reference[:n]

    mu_e = est.mean(axis=0)
    mu_r = ref.mean(axis=0)
    est_c = est - mu_e
    ref_c = ref - mu_r

    H = est_c.T @ ref_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1] * (est.shape[1] - 1) + [d])
    R = Vt.T @ D @ U.T
    t = mu_r - R @ mu_e

    return (R @ est.T).T + t


def ate(estimated: np.ndarray, reference: np.ndarray) -> float:
    """
    Absolute Trajectory Error — RMSE of per-frame position errors
    after aligning estimated to reference.
    """
    n = min(len(estimated), len(reference))
    aligned = align_trajectories(estimated, reference)
    diff = aligned[:n] - reference[:n]
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def rte(estimated: np.ndarray, reference: np.ndarray,
        segment_len: int = 100) -> float:
    """
    Relative Trajectory Error — average drift over fixed-length segments.
    Measures local consistency independent of global alignment.
    """
    n = min(len(estimated), len(reference))
    errors = []
    for i in range(0, n - segment_len, segment_len // 2):
        j = i + segment_len
        if j >= n:
            break
        # Relative motion in reference
        ref_rel = reference[j] - reference[i]
        est_rel = estimated[j] - estimated[i]
        errors.append(np.linalg.norm(ref_rel - est_rel))
    return float(np.mean(errors)) if errors else 0.0


def final_drift(estimated: np.ndarray, reference: np.ndarray) -> float:
    """Euclidean distance between final estimated and reference positions."""
    n = min(len(estimated), len(reference))
    return float(np.linalg.norm(estimated[n - 1] - reference[n - 1]))


# --- Plotting ---

def plot_evaluation(trajectories: dict, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Top-down trajectory comparison
    ax = axes[0]
    styles = {"GPS (reference)": ("black", "--", 2),
              "VO only":         ("steelblue", "-", 1),
              "EKF fused":       ("seagreen", "-", 2)}
    for label, traj in trajectories.items():
        c, ls, lw = styles.get(label, ("grey", "-", 1))
        ax.plot(traj[:, 0], traj[:, 1], label=label,
                color=c, linestyle=ls, linewidth=lw)
    ax.scatter([0], [0], color="black", zorder=5, s=50, label="Start")
    ax.set_title("Trajectory comparison (top-down)")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.legend()
    ax.grid(True)
    ax.set_aspect("equal")

    # Per-frame position error
    ax = axes[1]
    ref = trajectories.get("GPS (reference)")
    if ref is not None:
        for label, traj in trajectories.items():
            if label == "GPS (reference)":
                continue
            n = min(len(traj), len(ref))
            aligned = align_trajectories(traj, ref)
            errors = np.linalg.norm(aligned[:n] - ref[:n], axis=1)
            c, _, lw = styles.get(label, ("grey", "-", 1))
            ax.plot(errors, label=label, color=c, linewidth=lw)
    ax.set_title("Per-frame position error vs GPS")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Error (m)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()


# --- Main ---

if __name__ == "__main__":
    import sys, argparse
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.preprocessing.ingest import KITTIRawLoader
    from src.fusion.run_fusion import run as run_fusion

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  default="data/kitti/raw")
    parser.add_argument("--date",       default="2011_09_26")
    parser.add_argument("--drive",      default="2011_09_26_drive_0001")
    parser.add_argument("--max_frames", type=int, default=108)
    args = parser.parse_args()

    print("Running fusion pipeline...")
    vo_traj, gps_traj, ekf_traj, _ = run_fusion(
        args.data_path, args.date, args.drive, args.max_frames
    )

    # GPS East/North is our 2D ground truth reference
    ref = gps_traj[:, :2]   # (N, 2) East, North

    # VO is in camera frame — align to GPS via Umeyama (handles rotation+scale)
    vo  = vo_traj[:, [0, 2]]   # X, Z from camera frame
    ekf = ekf_traj[:, :2]      # EKF X, Y (already in GPS frame)

    # Align VO to GPS frame for fair comparison
    vo_aligned  = align_trajectories(vo, ref)
    ekf_aligned = align_trajectories(ekf, ref)

    n = min(len(vo_aligned), len(ekf_aligned), len(ref))

    def per_frame_errors(aligned, reference):
        nn = min(len(aligned), len(reference))
        return np.linalg.norm(aligned[:nn] - reference[:nn], axis=1)

    vo_errors  = per_frame_errors(vo_aligned, ref)
    ekf_errors = per_frame_errors(ekf_aligned, ref)

    print("\n--- Evaluation Results (vs GPS ground truth) ---")
    print(f"{'Metric':<30} {'VO only':>12} {'EKF fused':>12}")
    print("-" * 56)
    print(f"{'ATE (m)':<30} {np.sqrt(np.mean(vo_errors**2)):>12.3f} {np.sqrt(np.mean(ekf_errors**2)):>12.3f}")
    print(f"{'Mean error (m)':<30} {vo_errors.mean():>12.3f} {ekf_errors.mean():>12.3f}")
    print(f"{'Max error (m)':<30} {vo_errors.max():>12.3f} {ekf_errors.max():>12.3f}")
    print(f"{'Final drift (m)':<30} {vo_errors[-1]:>12.3f} {ekf_errors[-1]:>12.3f}")
    print("-" * 56)
    print("Note: VO evaluated after Umeyama alignment to GPS frame")

    trajectories = {
        "GPS (reference)": ref,
        "VO only":         vo_aligned,
        "EKF fused":       ekf_aligned,
    }
    plot_evaluation(trajectories, "results/evaluation.png")
