"""
FusionKit — single entry point.
Usage: python run.py --config configs/kitti.yaml
"""

import sys
import argparse
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_loader(cfg: dict):
    ds = cfg["dataset"]
    if ds["type"] == "kitti_raw":
        from src.preprocessing.ingest import KITTIRawLoader
        return KITTIRawLoader(ds["data_path"], ds["date"], ds["drive"])
    raise ValueError(f"Unknown dataset type: {ds['type']}")


def run_pipeline(cfg: dict):
    from src.fusion.run_fusion import run as run_fusion, plot as plot_fusion
    from src.reconstruction.build_map import accumulate_map, imu_pose
    from src.evaluation.evaluate import ate, rte, final_drift, align_trajectories, plot_evaluation

    ds      = cfg["dataset"]
    vis     = cfg.get("visualize", {})
    out     = cfg.get("output", {})
    max_f   = ds.get("max_frames", 108)

    # ── Step 1: Fusion (VO + GPS + IMU → trajectories) ──────────────────────
    print("\n[1/4] Running sensor fusion...")
    vo_traj, gps_traj, ekf_traj, vo_poses = run_fusion(
        ds["data_path"], ds["date"], ds["drive"], max_f
    )

    # Save trajectory
    traj_path = out.get("trajectory", "results/trajectory.txt")
    Path(traj_path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(traj_path, np.array([p.flatten() for p in vo_poses]))
    print(f"    Trajectory saved: {traj_path}")

    # Visualize fusion trajectories
    if vis.get("fusion_trajectory", True):
        plots_dir = out.get("plots_dir", "results/")
        Path(plots_dir).mkdir(parents=True, exist_ok=True)
        plot_fusion(vo_traj, gps_traj, ekf_traj)

    # ── Step 2: Evaluation ───────────────────────────────────────────────────
    eval_cfg = cfg.get("evaluation", {})
    if eval_cfg.get("enabled", True):
        print("\n[2/4] Running evaluation...")
        ref        = gps_traj[:, :2]
        vo_2d      = vo_traj[:, [0, 2]]
        ekf_2d     = ekf_traj[:, :2]
        vo_aligned  = align_trajectories(vo_2d, ref)
        ekf_aligned = align_trajectories(ekf_2d, ref)

        metrics = eval_cfg.get("metrics", ["ATE", "RTE", "drift"])
        n = min(len(vo_aligned), len(ekf_aligned), len(ref))
        vo_err  = np.linalg.norm(vo_aligned[:n]  - ref[:n],  axis=1)
        ekf_err = np.linalg.norm(ekf_aligned[:n] - ref[:n], axis=1)

        print(f"\n    {'Metric':<25} {'VO only':>10} {'EKF fused':>10}")
        print("    " + "-" * 47)
        if "ATE" in metrics:
            print(f"    {'ATE (m)':<25} {np.sqrt(np.mean(vo_err**2)):>10.3f} {np.sqrt(np.mean(ekf_err**2)):>10.3f}")
        if "RTE" in metrics:
            print(f"    {'RTE (m/segment)':<25} {rte(vo_2d, ref):>10.3f} {rte(ekf_2d, ref):>10.3f}")
        if "drift" in metrics:
            print(f"    {'Final drift (m)':<25} {vo_err[-1]:>10.3f} {ekf_err[-1]:>10.3f}")
        print("    " + "-" * 47)

        if vis.get("evaluation", True):
            plot_evaluation(
                {"GPS (reference)": ref, "VO only": vo_aligned, "EKF fused": ekf_aligned},
                str(Path(out.get("plots_dir", "results/")) / "evaluation.png")
            )

    # ── Step 3: 3D Reconstruction ────────────────────────────────────────────
    recon_cfg = cfg.get("reconstruction", {})
    if recon_cfg.get("enabled", True):
        print("\n[3/4] Building 3D map...")
        loader = build_loader(cfg)
        calib  = loader.load_calib()
        oxts0  = loader.load_oxts(0)
        lat0, lon0 = oxts0["lat"], oxts0["lon"]

        poses = []
        for i in range(min(max_f, len(loader))):
            oxts = loader.load_oxts(i)
            poses.append(imu_pose(oxts, oxts0, lat0, lon0))

        import open3d as o3d
        global_map = accumulate_map(
            loader, poses, calib, max_f,
            voxel_size=recon_cfg.get("voxel_size", 0.2)
        )

        map_path = out.get("map", "results/map.pcd")
        Path(map_path).parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(map_path, global_map)
        print(f"    Map saved: {map_path} ({len(global_map.points)} points)")

        # Bird's eye view
        if vis.get("birdseye", True):
            import matplotlib.pyplot as plt
            pts  = np.asarray(global_map.points)
            cols = np.asarray(global_map.colors)
            idx  = np.random.choice(len(pts), min(50000, len(pts)), replace=False)
            plt.figure(figsize=(14, 6))
            plt.scatter(pts[idx, 0], pts[idx, 2], c=cols[idx], s=0.3)
            plt.title("Bird's eye view (top-down)")
            plt.xlabel("X / East (m)")
            plt.ylabel("Z / North (m)")
            plt.axis("equal")
            plt.tight_layout()
            bev_path = str(Path(out.get("plots_dir", "results/")) / "map_birdseye.png")
            plt.savefig(bev_path, dpi=150)
            print(f"    Bird's eye view saved: {bev_path}")
            plt.show()

        # 3D viewer
        if vis.get("point_cloud", True):
            print("    Opening 3D viewer... (close window to continue)")
            o3d.visualization.draw_geometries([global_map], window_name="FusionKit — Global Map")

    # ── Step 4: Done ─────────────────────────────────────────────────────────
    print("\n[4/4] Done. Results saved to:", out.get("plots_dir", "results/"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FusionKit pipeline runner")
    parser.add_argument("--config", default="configs/kitti.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"FusionKit — loaded config: {args.config}")
    print(f"  Dataset : {cfg['dataset']['type']} / {cfg['dataset'].get('drive', '')}")
    print(f"  Frames  : {cfg['dataset'].get('max_frames', 'all')}")

    run_pipeline(cfg)
