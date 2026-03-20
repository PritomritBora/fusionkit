"""
Smoke test for NuScenesLoader — run this before the full pipeline
to verify data layout, calibration, and pose encoding are correct.

Usage:
    python sensor-fusion-3d/scripts/test_nuscenes_loader.py
    python sensor-fusion-3d/scripts/test_nuscenes_loader.py --dataroot data/nuscenes --scene 1
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io.nuscenes_loader import NuScenesLoader
from src.fusion.run_fusion import latlon_to_xy


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    return condition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version",  default="v1.0-mini")
    parser.add_argument("--scene",    type=int, default=0)
    args = parser.parse_args()

    print(f"\nNuScenesLoader smoke test")
    print(f"  dataroot : {args.dataroot}")
    print(f"  version  : {args.version}")
    print(f"  scene    : {args.scene}")
    print()

    # ── Load ────────────────────────────────────────────────────────────────
    try:
        loader = NuScenesLoader(args.dataroot, args.version, args.scene)
    except Exception as e:
        print(f"  [FAIL] Could not load dataset: {e}")
        print("\n  Make sure nuScenes mini is extracted to data/nuscenes/")
        print("  See: bash sensor-fusion-3d/scripts/download_nuscenes.sh")
        sys.exit(1)

    n = len(loader)
    print(f"Scene: {loader.scene_name()}  ({n} keyframes)")
    print(f"All scenes: {loader.list_scenes()}\n")

    all_pass = True

    # ── Frame count ─────────────────────────────────────────────────────────
    print("1. Frame count")
    all_pass &= check("len(loader) > 0", n > 0, f"{n} frames")
    all_pass &= check("Reasonable frame count (10–50)", 10 <= n <= 50, f"{n}")
    print()

    # ── Image ───────────────────────────────────────────────────────────────
    print("2. Image loading")
    img = loader.load_image(0)
    all_pass &= check("load_image returns ndarray", isinstance(img, np.ndarray))
    all_pass &= check("Image is 3-channel", img.ndim == 3 and img.shape[2] == 3,
                      f"shape={img.shape}")
    all_pass &= check("Image is ~1600x900", img.shape[1] > 1000 and img.shape[0] > 500,
                      f"{img.shape[1]}x{img.shape[0]}")
    print()

    # ── LiDAR ───────────────────────────────────────────────────────────────
    print("3. LiDAR loading")
    pts = loader.load_lidar(0)
    all_pass &= check("load_lidar returns ndarray", isinstance(pts, np.ndarray))
    all_pass &= check("LiDAR shape is (N, 4)", pts.ndim == 2 and pts.shape[1] == 4,
                      f"shape={pts.shape}")
    all_pass &= check("LiDAR has >1000 points", pts.shape[0] > 1000, f"{pts.shape[0]} pts")
    all_pass &= check("LiDAR dtype is float32", pts.dtype == np.float32)
    print()

    # ── Calibration ─────────────────────────────────────────────────────────
    print("4. Calibration")
    calib = loader.load_calib()
    all_pass &= check("P2 shape (3,4)", calib["P2"].shape == (3, 4))
    all_pass &= check("R0_rect shape (3,3)", calib["R0_rect"].shape == (3, 3))
    all_pass &= check("Tr_velo_to_cam shape (3,4)", calib["Tr_velo_to_cam"].shape == (3, 4))
    all_pass &= check("imu_to_velo shape (3,4)", calib["imu_to_velo"].shape == (3, 4))

    # Focal length sanity (nuScenes CAM_FRONT fx ≈ 1266)
    fx = calib["P2"][0, 0]
    all_pass &= check("Focal length fx > 500", fx > 500, f"fx={fx:.1f}")
    print()

    # ── Pose hints ──────────────────────────────────────────────────────────
    print("5. Pose hints (lat/lon encoding)")
    hint0 = loader.load_pose_hint(0)
    all_pass &= check("Frame 0 lat == 0.0", abs(hint0["lat"]) < 1e-9,
                      f"lat={hint0['lat']:.2e}")
    all_pass &= check("Frame 0 lon == 0.0", abs(hint0["lon"]) < 1e-9,
                      f"lon={hint0['lon']:.2e}")
    all_pass &= check("Frame 0 alt == 0.0", abs(hint0["alt"]) < 1e-6,
                      f"alt={hint0['alt']:.2e}")

    # Collect all poses and verify round-trip
    lats = [loader.load_pose_hint(i)["lat"] for i in range(n)]
    lons = [loader.load_pose_hint(i)["lon"] for i in range(n)]
    xs, ys = zip(*[latlon_to_xy(lat, lon, 0.0, 0.0) for lat, lon in zip(lats, lons)])
    xs, ys = np.array(xs), np.array(ys)

    total_dist = np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2))
    max_disp   = np.max(np.sqrt(xs**2 + ys**2))

    all_pass &= check("Trajectory has non-zero displacement", total_dist > 1.0,
                      f"total path={total_dist:.1f}m")
    all_pass &= check("Max displacement < 2000m (sanity)", max_disp < 2000,
                      f"max={max_disp:.1f}m")

    print(f"  Trajectory: {total_dist:.1f}m total path, {max_disp:.1f}m max displacement")
    print()

    # ── Timestamps ──────────────────────────────────────────────────────────
    print("6. Timestamps")
    ts = loader.load_timestamps()
    all_pass &= check("Timestamps shape matches frames", len(ts) == n, f"{len(ts)} vs {n}")
    dt = np.diff(ts)
    all_pass &= check("Frame interval ~0.5s (2 Hz)", 0.3 < dt.mean() < 1.0,
                      f"mean dt={dt.mean():.3f}s")
    print()

    # ── LiDAR projection sanity ─────────────────────────────────────────────
    print("7. LiDAR → camera projection")
    from src.calibration.calibrate import lidar_to_camera
    cam_pts, pixels, mask = lidar_to_camera(pts, calib)
    h, w = img.shape[:2]
    u, v = pixels[:, 0], pixels[:, 1]
    in_fov = mask & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    fov_pct = in_fov.sum() / len(pts) * 100
    all_pass &= check("Some points project into camera FOV", in_fov.sum() > 100,
                      f"{in_fov.sum()} pts ({fov_pct:.1f}%)")
    all_pass &= check("FOV coverage 5–40%", 5 < fov_pct < 40, f"{fov_pct:.1f}%")
    print()

    # ── Summary ─────────────────────────────────────────────────────────────
    print("=" * 50)
    if all_pass:
        print("All checks passed. Ready to run:")
        print("  python sensor-fusion-3d/run.py --config sensor-fusion-3d/configs/nuscenes.yaml")
    else:
        print("Some checks FAILED — review output above before running the pipeline.")
    print("=" * 50)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
