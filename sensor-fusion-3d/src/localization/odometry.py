"""
Stage 3 — Visual Odometry
Estimate camera trajectory using ORB feature matching and pose estimation.
Optionally fuse with IMU via a simple EKF.
"""

import numpy as np
import cv2
from pathlib import Path


class VisualOdometry:
    """Frame-to-frame VO using ORB features and essential matrix decomposition."""

    def __init__(self, K: np.ndarray):
        self.K = K  # (3, 3) intrinsic matrix
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None

        # Accumulated pose: list of (3, 4) matrices
        self.poses = [np.eye(3, 4)]

    def _extract(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        return kp, des

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process next frame, return current pose (3, 4)."""
        kp, des = self._extract(frame)

        if self.prev_frame is None:
            self.prev_kp, self.prev_des = kp, des
            self.prev_frame = frame
            return self.poses[-1]

        matches = self.matcher.match(self.prev_des, des)
        matches = sorted(matches, key=lambda m: m.distance)[:500]

        pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(pts1, pts2, self.K,
                                        method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)

        # Compose with previous pose (camera-to-world)
        prev_pose = self.poses[-1]
        R_prev = prev_pose[:, :3]
        t_prev = prev_pose[:, 3:]

        R_new = R_prev @ R.T
        t_new = t_prev - R_new @ t

        new_pose = np.hstack([R_new, t_new])
        self.poses.append(new_pose)

        self.prev_kp, self.prev_des = kp, des
        self.prev_frame = frame

        return new_pose


def extract_intrinsics(calib: dict) -> np.ndarray:
    """Extract 3x3 K matrix from KITTI P2."""
    P2 = calib["P2"]
    return P2[:3, :3]


def save_trajectory(poses: list, out_path: str):
    """Save trajectory as (N, 12) KITTI-format pose file."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    arr = np.array([p.flatten() for p in poses])
    np.savetxt(out_path, arr)
    print(f"Trajectory saved: {out_path} ({len(poses)} poses)")


if __name__ == "__main__":
    import sys, argparse
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tqdm import tqdm
    from src.preprocessing.ingest import KITTIRawLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/kitti/raw")
    parser.add_argument("--date", default="2011_09_26")
    parser.add_argument("--drive", default="2011_09_26_drive_0001")
    parser.add_argument("--max_frames", type=int, default=108)
    args = parser.parse_args()

    loader = KITTIRawLoader(args.data_path, args.date, args.drive)
    calib = loader.load_calib()
    K = extract_intrinsics(calib)

    vo = VisualOdometry(K)
    n = min(args.max_frames, len(loader))

    for i in tqdm(range(n), desc="Visual Odometry"):
        frame = loader.load_image(i)
        vo.process_frame(frame)

    save_trajectory(vo.poses, f"results/trajectory_{args.drive}.txt")
