"""
Stage 5 — Enhancement: Mesh Reconstruction
Convert point cloud to mesh using Poisson reconstruction,
then export as .obj for Unreal Engine import.
"""

import open3d as o3d
import numpy as np
from pathlib import Path


def poisson_mesh(pcd: o3d.geometry.PointCloud,
                 depth: int = 9) -> o3d.geometry.TriangleMesh:
    """Run Poisson surface reconstruction. Requires normals on pcd."""
    print("Running Poisson reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # Remove low-density vertices (artifacts at boundaries)
    density_threshold = np.quantile(np.asarray(densities), 0.05)
    vertices_to_remove = np.asarray(densities) < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()
    return mesh


def export_mesh(mesh: o3d.geometry.TriangleMesh, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f"Mesh exported: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd", default="results/map_seq00.pcd")
    parser.add_argument("--out", default="unreal/mesh/scene.obj")
    parser.add_argument("--depth", type=int, default=9)
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.pcd)
    print(f"Loaded {len(pcd.points)} points")

    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
        )

    mesh = poisson_mesh(pcd, depth=args.depth)
    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    export_mesh(mesh, args.out)
