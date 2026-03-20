"""
Stage 4 — 3D Reconstruction
Accumulate LiDAR frames into a global point cloud using ground truth or
estimated poses. Applies ICP refinement between consecutive frames.
"""

import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm


def load_poses(pose_file: str) -> list:
    """Load (N, 3, 4) poses from KITTI-format file."""
    raw = np.loadtxt(pose_file)
    poses = []
    for row in raw:
        T = np.eye(4)
        T[:3, :] = row.reshape(3, 4)
        poses.append(T)
    return poses


def lidar_to_open3d(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd


def imu_pose(oxts: dict, oxts0: dict, lat0: float, lon0: float) -> np.ndarray:
    """Build 4x4 world pose from GPS position + IMU orientation."""
    from src.fusion.run_fusion import latlon_to_xy

    # Translation from GPS
    x, z = latlon_to_xy(oxts["lat"], oxts["lon"], lat0, lon0)
    y = oxts["alt"] - oxts0["alt"]

    # Rotation from IMU roll/pitch/yaw (ZYX convention)
    r, p, yaw = oxts["roll"], oxts["pitch"], oxts["yaw"]
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0,            0,           1]])
    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [0,          1, 0         ],
                   [-np.sin(p), 0, np.cos(p)]])
    Rx = np.array([[1, 0,           0          ],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r),  np.cos(r)]])
    R = Rz @ Ry @ Rx

    T = np.eye(4)
    T[:3, :3] = R
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


def accumulate_map(loader, poses: list, calib: dict, max_frames: int = 100,
                   voxel_size: float = 0.2) -> o3d.geometry.PointCloud:
    """
    Accumulate RGB-colored LiDAR frames into a single global point cloud.

    For KITTI: poses are IMU world-frame transforms, chain is LiDAR→IMU→world.
    For nuScenes: if loader has load_ego_poses(), uses exact ego quaternion
                  transforms and chain is LiDAR→ego→world.
    """
    from src.calibration.calibrate import lidar_to_camera

    # Use exact ego poses if the loader provides them (nuScenes)
    if hasattr(loader, 'load_ego_poses'):
        ego_poses = loader.load_ego_poses()
        use_ego_poses = True
    else:
        ego_poses = None
        use_ego_poses = False

    # LiDAR-to-ego transform (T_ego_from_lidar) for nuScenes
    # LiDAR-to-IMU transform (velo_to_imu) for KITTI
    imu_to_velo = calib.get("imu_to_velo")  # (3,4)
    if imu_to_velo is not None:
        Tv = np.eye(4)
        Tv[:3, :] = imu_to_velo
        if use_ego_poses:
            # For nuScenes: imu_to_velo = inv(T_ego_from_lidar), so
            # T_ego_from_lidar = inv(imu_to_velo)
            T_ego_from_lidar = np.linalg.inv(Tv)
        else:
            # For KITTI: velo_to_imu = inv(imu_to_velo)
            T_ego_from_lidar = np.linalg.inv(Tv)
    else:
        T_ego_from_lidar = np.eye(4)

    global_pcd = o3d.geometry.PointCloud()

    n = min(max_frames, len(loader), len(ego_poses) if use_ego_poses else len(poses))
    for i in tqdm(range(n), desc="Building map"):
        pts = loader.load_lidar(i)
        img = loader.load_image(i)
        h, w = img.shape[:2]

        # Project to get RGB for in-FOV points
        _, pixels, mask = lidar_to_camera(pts, calib)
        u, v = pixels[:, 0], pixels[:, 1]
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        colored = mask & in_bounds

        colors = np.full((len(pts), 3), 0.5)
        colors[colored] = img[v[colored], u[colored]][:, ::-1] / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[colored, :3])
        pcd.colors = o3d.utility.Vector3dVector(colors[colored])

        # Apply world transform: LiDAR → ego/IMU → world
        world_pose = ego_poses[i] if use_ego_poses else poses[i]
        pcd.transform(world_pose @ T_ego_from_lidar)
        global_pcd += pcd

        if i % 10 == 0:
            global_pcd = global_pcd.voxel_down_sample(voxel_size)

    global_pcd = global_pcd.voxel_down_sample(voxel_size)

    print("Removing outliers...")
    global_pcd, _ = global_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    print(f"Clean map: {len(global_pcd.points)} points")
    return global_pcd


if __name__ == "__main__":
    import sys, argparse
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.preprocessing.ingest import KITTIRawLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  default="data/kitti/raw")
    parser.add_argument("--date",       default="2011_09_26")
    parser.add_argument("--drive",      default="2011_09_26_drive_0001")
    parser.add_argument("--max_frames", type=int, default=108)
    parser.add_argument("--voxel_size", type=float, default=0.2)
    args = parser.parse_args()

    loader = KITTIRawLoader(args.data_path, args.date, args.drive)
    calib  = loader.load_calib()

    # Build poses from GPS + IMU (proper world-frame transforms)
    oxts0 = loader.load_oxts(0)
    lat0, lon0 = oxts0["lat"], oxts0["lon"]
    poses = []
    for i in range(min(args.max_frames, len(loader))):
        oxts = loader.load_oxts(i)
        poses.append(imu_pose(oxts, oxts0, lat0, lon0))

    global_map = accumulate_map(loader, poses, calib, args.max_frames, args.voxel_size)

    out_path = f"results/map_{args.drive}.pcd"
    Path("results").mkdir(exist_ok=True)
    o3d.io.write_point_cloud(out_path, global_map)
    print(f"Map saved: {out_path} ({len(global_map.points)} points)")

    # Bird's eye view plot (X vs Z top-down)
    import matplotlib.pyplot as plt
    pts = np.asarray(global_map.points)
    cols = np.asarray(global_map.colors)
    # Subsample for plotting speed
    idx = np.random.choice(len(pts), min(50000, len(pts)), replace=False)
    plt.figure(figsize=(14, 6))
    plt.scatter(pts[idx, 0], pts[idx, 2], c=cols[idx], s=0.3)
    plt.title("Bird's eye view (top-down, X=East, Z=North)")
    plt.xlabel("X / East (m)")
    plt.ylabel("Z / North (m)")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("results/map_birdseye.png", dpi=150)
    print("Saved: results/map_birdseye.png")
    plt.show()

    o3d.visualization.draw_geometries([global_map], window_name="Global Map")
