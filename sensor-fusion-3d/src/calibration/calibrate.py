"""
Stage 2 — Sensor Calibration
Parse KITTI calibration files and project LiDAR points into the camera frame.
Outputs a colored point cloud for visual verification.
"""

import numpy as np
import cv2
import open3d as o3d
from pathlib import Path


def load_kitti_calib(calib_path: str) -> dict:
    """Parse KITTI calib.txt into projection/transform matrices."""
    calib = {}
    with open(calib_path) as f:
        for line in f:
            key, *vals = line.strip().split()
            calib[key.rstrip(":")] = np.array(vals, dtype=np.float64)

    # Camera 2 projection matrix (3x4)
    calib["P2"] = calib["P2"].reshape(3, 4)
    # Rectification matrix (3x3)
    calib["R0_rect"] = calib["R0_rect"].reshape(3, 3)
    # LiDAR to camera transform (3x4)
    calib["Tr_velo_to_cam"] = calib["Tr_velo_to_cam"].reshape(3, 4)

    return calib


def lidar_to_camera(points: np.ndarray, calib: dict) -> tuple:
    """
    Project LiDAR points into camera image space.
    Returns (camera_pts, pixel_coords, valid_mask).
    """
    Tr = calib["Tr_velo_to_cam"]  # (3, 4)
    R0 = calib["R0_rect"]          # (3, 3)
    P2 = calib["P2"]               # (3, 4)

    # Homogeneous LiDAR points (N, 4)
    pts_hom = np.hstack([points[:, :3], np.ones((len(points), 1))])

    # To camera coords
    cam_pts = (R0 @ Tr @ pts_hom.T).T  # (N, 3)

    # Keep points in front of camera
    mask = cam_pts[:, 2] > 0

    # Project to image
    pts_hom_cam = np.hstack([cam_pts, np.ones((len(cam_pts), 1))])
    pixels = (P2 @ pts_hom_cam.T).T  # (N, 3)
    pixels[:, :2] /= pixels[:, 2:3]

    return cam_pts, pixels[:, :2].astype(int), mask


def color_point_cloud(points: np.ndarray, pixels: np.ndarray,
                      mask: np.ndarray, image: np.ndarray,
                      img_h: int, img_w: int) -> o3d.geometry.PointCloud:
    """Sample RGB colors from image for each projected LiDAR point."""
    u, v = pixels[:, 0], pixels[:, 1]
    in_bounds = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    valid = mask & in_bounds

    pts_valid = points[valid, :3]
    colors = image[v[valid], u[valid]] / 255.0  # BGR -> normalize
    colors = colors[:, ::-1]  # BGR to RGB

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_valid)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def depth_color_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    """Color all LiDAR points by depth (distance from sensor) for full-scan visualization."""
    pts = points[:, :3]
    depth = np.linalg.norm(pts, axis=1)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    # Map to jet-like colormap: blue=near, red=far
    colors = np.zeros((len(pts), 3))
    colors[:, 0] = depth_norm          # R
    colors[:, 2] = 1.0 - depth_norm    # B

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


if __name__ == "__main__":
    import sys, argparse
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.preprocessing.ingest import KITTIRawLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/kitti/raw")
    parser.add_argument("--date", default="2011_09_26")
    parser.add_argument("--drive", default="2011_09_26_drive_0001")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--mode", choices=["rgb", "depth"], default="depth",
                        help="rgb=camera-colored (FOV only), depth=all points colored by distance")
    args = parser.parse_args()

    loader = KITTIRawLoader(args.data_path, args.date, args.drive)
    calib = loader.load_calib()

    img = loader.load_image(args.frame)
    pts = loader.load_lidar(args.frame)

    h, w = img.shape[:2]
    cam_pts, pixels, mask = lidar_to_camera(pts, calib)

    if args.mode == "rgb":
        pcd = color_point_cloud(pts, pixels, mask, img, h, w)
        print(f"RGB-colored points (camera FOV only): {len(pcd.points)}")
    else:
        pcd = depth_color_point_cloud(pts)
        print(f"Depth-colored points (full 360° scan): {len(pcd.points)}")

    out_path = f"results/images/pcd_{args.mode}_frame{args.frame:06d}.pcd"
    Path("results/images").mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"Saved: {out_path}")

    o3d.visualization.draw_geometries([pcd], window_name=f"Point Cloud [{args.mode}]")
