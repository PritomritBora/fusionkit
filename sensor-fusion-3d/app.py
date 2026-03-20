"""
FusionKit — Streamlit frontend
Run: streamlit run sensor-fusion-3d/app.py
"""

import sys
import threading
import queue
import time
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FusionKit",
    page_icon="🗺️",
    layout="wide",
)

# ── constants ────────────────────────────────────────────────────────────────
KITTI_DATA   = "data/kitti/raw"
KITTI_DATE   = "2011_09_26"
KITTI_DRIVE  = "2011_09_26_drive_0001"
KITTI_FRAMES = 108

NUSCENES_ROOT    = "data/v1.0-mini"
NUSCENES_VERSION = "v1.0-mini"
NUSCENES_SCENES  = [
    "scene-0061",  # 0 — 92m
    "scene-0103",  # 1 — 118m
    "scene-0553",  # 2 — 0m  (stationary)
    "scene-0655",  # 3 — 163m
    "scene-0757",  # 4 — 28m
    "scene-0796",  # 5 — 237m
    "scene-0916",  # 6 — 93m
    "scene-1077",  # 7 — 252m
    "scene-1094",  # 8 — 126m
    "scene-1100",  # 9 — 1m   (stationary)
]
NUSCENES_SCENE_LABELS = [
    "scene-0061  (92m)",
    "scene-0103  (118m)",
    "scene-0553  (0m — stationary)",
    "scene-0655  (163m)",
    "scene-0757  (28m)",
    "scene-0796  (237m)",
    "scene-0916  (93m)",
    "scene-1077  (252m)",
    "scene-1094  (126m)",
    "scene-1100  (1m — stationary)",
]


# ── sidebar — dataset config ─────────────────────────────────────────────────
st.sidebar.title("🗺️ FusionKit")
st.sidebar.markdown("Sensor fusion & 3D mapping")
st.sidebar.divider()

dataset = st.sidebar.radio("Dataset", ["KITTI Raw", "nuScenes"], horizontal=True)

if dataset == "KITTI Raw":
    st.sidebar.markdown(f"**Drive:** `{KITTI_DRIVE}`")
    max_frames = st.sidebar.slider("Max frames", 10, KITTI_FRAMES, KITTI_FRAMES, step=10)
    scene_index = 0
else:
    scene_label = st.sidebar.selectbox("Scene", NUSCENES_SCENE_LABELS, index=7)
    scene_index = NUSCENES_SCENE_LABELS.index(scene_label)
    max_frames  = st.sidebar.slider("Max frames", 5, 41, 41, step=1)

st.sidebar.divider()
st.sidebar.markdown("**Pipeline settings**")
voxel_size = st.sidebar.slider("Voxel size (m)", 0.1, 1.0, 0.2, step=0.05)
run_recon  = st.sidebar.checkbox("3D reconstruction", value=True)
run_eval   = st.sidebar.checkbox("Evaluation", value=True)

st.sidebar.divider()
run_btn = st.sidebar.button("▶  Run Pipeline", type="primary", use_container_width=True)


# ── main area ────────────────────────────────────────────────────────────────
st.title("FusionKit — Sensor Fusion & 3D Mapping")

col_info, col_status = st.columns([2, 1])
with col_info:
    if dataset == "KITTI Raw":
        st.info(f"**KITTI Raw** · drive `{KITTI_DRIVE}` · {max_frames} frames · voxel {voxel_size}m")
    else:
        st.info(f"**nuScenes** · `{NUSCENES_SCENES[scene_index]}` · {max_frames} frames · voxel {voxel_size}m")

log_box    = st.empty()
metrics_ph = st.empty()
tabs_ph    = st.empty()


# ── pipeline runner ──────────────────────────────────────────────────────────

def run_pipeline(dataset, scene_index, max_frames, voxel_size, run_recon, run_eval, log_q):
    """Runs in a background thread. Puts log lines and result dicts into log_q."""
    try:
        from src.fusion.run_fusion import run as run_kitti, run_with_loader, latlon_to_xy
        from src.reconstruction.build_map import accumulate_map, imu_pose
        from src.evaluation.evaluate import ate, rte, align_trajectories
        from src.localization.odometry import extract_intrinsics

        def log(msg):
            log_q.put(("log", msg))

        # ── build loader ────────────────────────────────────────────────────
        log("Loading dataset...")
        if dataset == "KITTI Raw":
            from src.preprocessing.ingest import KITTIRawLoader
            loader = KITTIRawLoader(KITTI_DATA, KITTI_DATE, KITTI_DRIVE)
        else:
            from src.io.nuscenes_loader import NuScenesLoader
            loader = NuScenesLoader(NUSCENES_ROOT, NUSCENES_VERSION, scene_index)
        log(f"Loaded {len(loader)} frames")

        # ── fusion ──────────────────────────────────────────────────────────
        log("Running sensor fusion (VO + EKF)...")
        ekf_dt = 0.1 if dataset == "KITTI Raw" else 0.5

        if dataset == "KITTI Raw":
            vo_traj, gps_traj, ekf_traj, vo_poses = run_kitti(
                KITTI_DATA, KITTI_DATE, KITTI_DRIVE, max_frames
            )
        else:
            vo_traj, gps_traj, ekf_traj, vo_poses = run_with_loader(loader, max_frames, ekf_dt)

        log(f"Fusion done — {len(vo_traj)} frames")
        log_q.put(("trajectories", (vo_traj, gps_traj, ekf_traj)))

        # ── evaluation ──────────────────────────────────────────────────────
        if run_eval:
            log("Computing metrics...")
            from src.evaluation.evaluate import align_trajectories
            ref        = gps_traj[:, :2]
            vo_2d      = vo_traj[:, [0, 2]]
            ekf_2d     = ekf_traj[:, :2]
            vo_aligned  = align_trajectories(vo_2d,  ref)
            ekf_aligned = align_trajectories(ekf_2d, ref)
            n = min(len(vo_aligned), len(ekf_aligned), len(ref))
            vo_err  = np.linalg.norm(vo_aligned[:n]  - ref[:n],  axis=1)
            ekf_err = np.linalg.norm(ekf_aligned[:n] - ref[:n], axis=1)
            metrics = {
                "vo_ate":   float(np.sqrt(np.mean(vo_err**2))),
                "ekf_ate":  float(np.sqrt(np.mean(ekf_err**2))),
                "vo_drift":  float(vo_err[-1]),
                "ekf_drift": float(ekf_err[-1]),
            }
            log(f"ATE — VO: {metrics['vo_ate']:.2f}m  EKF: {metrics['ekf_ate']:.2f}m")
            log_q.put(("metrics", metrics))
            log_q.put(("eval_trajs", (ref, vo_aligned, ekf_aligned, vo_err, ekf_err)))

        # ── reconstruction ──────────────────────────────────────────────────
        if run_recon:
            log("Building 3D map...")
            calib = loader.load_calib()

            if hasattr(loader, 'load_ego_poses'):
                poses = loader.load_ego_poses()
            else:
                pose0      = loader.load_pose_hint(0)
                lat0, lon0 = pose0["lat"], pose0["lon"]
                poses = [imu_pose(loader.load_pose_hint(i), pose0, lat0, lon0)
                         for i in range(min(max_frames, len(loader)))]

            import open3d as o3d
            global_map = accumulate_map(loader, poses, calib, max_frames, voxel_size)
            pts  = np.asarray(global_map.points)
            cols = np.asarray(global_map.colors)
            log(f"Map built — {len(pts):,} points")
            log_q.put(("pointcloud", (pts, cols)))

        log("Done.")
        log_q.put(("done", None))

    except Exception as e:
        import traceback
        log_q.put(("error", traceback.format_exc()))


# ── plotly helpers ───────────────────────────────────────────────────────────

def trajectory_figure(vo_traj, gps_traj, ekf_traj):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gps_traj[:, 0], y=gps_traj[:, 1],
                             mode="lines", name="GPS reference",
                             line=dict(color="tomato", dash="dash", width=2)))
    fig.add_trace(go.Scatter(x=vo_traj[:, 0], y=vo_traj[:, 2],
                             mode="lines", name="VO only",
                             line=dict(color="steelblue", width=1)))
    fig.add_trace(go.Scatter(x=ekf_traj[:, 0], y=ekf_traj[:, 1],
                             mode="lines", name="EKF fused",
                             line=dict(color="seagreen", width=2)))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="Start",
                             marker=dict(color="black", size=10, symbol="circle")))
    fig.update_layout(
        title="Trajectory (top-down)",
        xaxis_title="East (m)", yaxis_title="North (m)",
        yaxis=dict(scaleanchor="x"),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=0, r=0, t=40, b=0),
        height=420,
    )
    return fig


def pointcloud_figure(pts, cols):
    # Subsample for browser performance
    n = min(40_000, len(pts))
    idx = np.random.choice(len(pts), n, replace=False)
    p, c = pts[idx], cols[idx]
    colors = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in c]
    fig = go.Figure(go.Scatter3d(
        x=p[:, 0], y=p[:, 2], z=p[:, 1],   # X=East, Y=North, Z=Up
        mode="markers",
        marker=dict(size=1.2, color=colors, opacity=0.85),
    ))
    fig.update_layout(
        title=f"3D Map ({n:,} pts shown)",
        scene=dict(
            xaxis_title="East (m)",
            yaxis_title="North (m)",
            zaxis_title="Up (m)",
            aspectmode="data",
            bgcolor="rgb(15,15,20)",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=520,
        paper_bgcolor="rgb(15,15,20)",
        font_color="white",
    )
    return fig


def evaluation_figure(ref, vo_aligned, ekf_aligned, vo_err, ekf_err):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Aligned trajectories vs GPS",
                                        "Per-frame position error"))
    fig.add_trace(go.Scatter(x=ref[:, 0], y=ref[:, 1], mode="lines",
                             name="GPS reference",
                             line=dict(color="tomato", dash="dash", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=vo_aligned[:, 0], y=vo_aligned[:, 1], mode="lines",
                             name="VO aligned",
                             line=dict(color="steelblue", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ekf_aligned[:, 0], y=ekf_aligned[:, 1], mode="lines",
                             name="EKF aligned",
                             line=dict(color="seagreen", width=2)), row=1, col=1)
    frames = list(range(len(vo_err)))
    fig.add_trace(go.Scatter(x=frames, y=vo_err.tolist(), mode="lines",
                             name="VO error", showlegend=False,
                             line=dict(color="steelblue", width=1)), row=1, col=2)
    fig.add_trace(go.Scatter(x=frames, y=ekf_err.tolist(), mode="lines",
                             name="EKF error", showlegend=False,
                             line=dict(color="seagreen", width=2)), row=1, col=2)
    fig.add_hline(y=float(np.sqrt(np.mean(vo_err**2))),
                  line=dict(color="steelblue", dash="dot", width=1),
                  annotation_text="VO ATE", row=1, col=2)
    fig.add_hline(y=float(np.sqrt(np.mean(ekf_err**2))),
                  line=dict(color="seagreen", dash="dot", width=1),
                  annotation_text="EKF ATE", row=1, col=2)
    fig.update_xaxes(title_text="East (m)", row=1, col=1)
    fig.update_yaxes(title_text="North (m)", scaleanchor="x", row=1, col=1)
    fig.update_xaxes(title_text="Frame", row=1, col=2)
    fig.update_yaxes(title_text="Error (m)", row=1, col=2)
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=40, b=0),
                      legend=dict(orientation="h", y=-0.15))
    return fig


def birdseye_figure(pts, cols):
    n = min(60_000, len(pts))
    idx = np.random.choice(len(pts), n, replace=False)
    p, c = pts[idx], cols[idx]
    colors = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in c]
    fig = go.Figure(go.Scatter(
        x=p[:, 0], y=p[:, 2],
        mode="markers",
        marker=dict(size=1.5, color=colors, opacity=0.9),
    ))
    fig.update_layout(
        title=f"Bird's eye view — top-down ({n:,} pts)",
        xaxis_title="East (m)",
        yaxis_title="North (m)",
        yaxis=dict(scaleanchor="x"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
        plot_bgcolor="rgb(10,10,15)",
        paper_bgcolor="rgb(10,10,15)",
        font_color="white",
    )
    return fig


# ── run on button press ──────────────────────────────────────────────────────

if run_btn:
    log_q: queue.Queue = queue.Queue()

    t = threading.Thread(
        target=run_pipeline,
        args=(dataset, scene_index, max_frames, voxel_size, run_recon, run_eval, log_q),
        daemon=True,
    )
    t.start()

    logs      = []
    metrics   = None
    trajs     = None
    pcd_data  = None
    eval_data = None
    done      = False

    log_area = log_box.empty()

    while not done:
        try:
            kind, payload = log_q.get(timeout=0.2)
        except queue.Empty:
            log_area.code("\n".join(logs) if logs else "Starting...", language=None)
            continue

        if kind == "log":
            logs.append(f"  {payload}")
            log_area.code("\n".join(logs), language=None)
        elif kind == "trajectories":
            trajs = payload
        elif kind == "metrics":
            metrics = payload
        elif kind == "eval_trajs":
            eval_data = payload
        elif kind == "pointcloud":
            pcd_data = payload
        elif kind == "done":
            done = True
        elif kind == "error":
            st.error(f"Pipeline error:\n\n```\n{payload}\n```")
            done = True

        time.sleep(0.05)

    log_area.code("\n".join(logs), language=None)

    # ── metrics row ─────────────────────────────────────────────────────────
    if metrics:
        with metrics_ph.container():
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("VO — ATE",    f"{metrics['vo_ate']:.2f} m")
            c2.metric("EKF — ATE",   f"{metrics['ekf_ate']:.2f} m",
                      delta=f"{metrics['ekf_ate'] - metrics['vo_ate']:.2f} m",
                      delta_color="inverse")
            c3.metric("VO — Drift",  f"{metrics['vo_drift']:.2f} m")
            c4.metric("EKF — Drift", f"{metrics['ekf_drift']:.2f} m",
                      delta=f"{metrics['ekf_drift'] - metrics['vo_drift']:.2f} m",
                      delta_color="inverse")

    # ── result tabs ──────────────────────────────────────────────────────────
    if trajs or pcd_data:
        with tabs_ph.container():
            st.divider()
            tabs = st.tabs(["📍 Trajectory", "📊 Evaluation", "🛰️ Bird's Eye", "🗺️ 3D Map"])

            with tabs[0]:
                if trajs:
                    st.plotly_chart(trajectory_figure(*trajs), use_container_width=True)
                else:
                    st.info("Fusion not run.")

            with tabs[1]:
                if eval_data:
                    st.plotly_chart(evaluation_figure(*eval_data), use_container_width=True)
                else:
                    st.info("Evaluation disabled.")

            with tabs[2]:
                if pcd_data:
                    st.plotly_chart(birdseye_figure(*pcd_data), use_container_width=True)
                else:
                    st.info("Reconstruction disabled.")

            with tabs[3]:
                if pcd_data:
                    st.plotly_chart(pointcloud_figure(*pcd_data), use_container_width=True)
                else:
                    st.info("Reconstruction disabled.")
