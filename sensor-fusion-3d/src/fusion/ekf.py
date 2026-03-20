"""
Stage 3 (Fusion) — Extended Kalman Filter
Fuses visual odometry pose estimates with IMU acceleration data.
State vector: [x, y, z, vx, vy, vz] (position + velocity)
"""

import numpy as np


class EKF:
    """Simple 6-DOF EKF for fusing VO position with IMU acceleration."""

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.n = 6  # state dim

        # State: [x, y, z, vx, vy, vz]
        self.x = np.zeros(self.n)

        # State covariance
        self.P = np.eye(self.n) * 0.1

        # Process noise
        self.Q = np.eye(self.n) * 0.01

        # Measurement noise (VO position)
        self.R_vo = np.eye(3) * 0.5

        # State transition
        self.F = np.eye(self.n)
        self.F[:3, 3:] = np.eye(3) * dt

        # VO observation matrix (measures position only)
        self.H_vo = np.zeros((3, self.n))
        self.H_vo[:3, :3] = np.eye(3)

    def predict(self, accel: np.ndarray):
        """Predict step using IMU acceleration."""
        # Control input: integrate acceleration into velocity
        B = np.zeros((self.n, 3))
        B[3:, :] = np.eye(3) * self.dt

        self.x = self.F @ self.x + B @ accel
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_vo(self, position: np.ndarray):
        """Update step using VO position measurement."""
        y = position - self.H_vo @ self.x
        S = self.H_vo @ self.P @ self.H_vo.T + self.R_vo
        K = self.P @ self.H_vo.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(self.n) - K @ self.H_vo) @ self.P

    @property
    def position(self) -> np.ndarray:
        return self.x[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:]
