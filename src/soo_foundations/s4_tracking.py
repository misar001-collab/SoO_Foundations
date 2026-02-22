#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 4 temporal tracking utilities (2D constant-velocity Kalman filter).

This module smooths a sequence of noisy 2D position measurements over time.
It is intended as the first Stage 4 baseline after Stage 3 positioning:
- Stage 3 gives per-frame position estimates.
- Stage 4 tracks those estimates across frames to reduce jitter and improve
  motion consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TrackingConfig2D:
    """
    Tunable settings for the 2D constant-velocity Kalman tracker.

    Meanings:
    - `dt_s`: frame interval in seconds.
    - `process_accel_sigma_mps2`: model uncertainty in acceleration.
    - `measurement_sigma_m`: standard deviation of position measurements.
    - `init_pos_sigma_m`: initial position uncertainty.
    - `init_vel_sigma_mps`: initial velocity uncertainty.
    """

    dt_s: float = 1.0
    process_accel_sigma_mps2: float = 1.0
    measurement_sigma_m: float = 10.0
    init_pos_sigma_m: float = 200.0
    init_vel_sigma_mps: float = 80.0


@dataclass(frozen=True)
class TrackingPoint2D:
    """One time-step tracking output bundle."""

    step: int
    measured_xy: Tuple[float, float]
    predicted_xy: Tuple[float, float]
    filtered_xy: Tuple[float, float]
    innovation_xy: Tuple[float, float]


def _state_matrices(dt_s: float, accel_sigma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build linear state-space matrices for constant-velocity motion in 2D.

    State vector:
      x = [px, py, vx, vy]^T
    Measurement:
      z = [px, py]^T
    """
    dt = float(dt_s)
    q = float(accel_sigma) ** 2

    f = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    # Process noise built from white acceleration model.
    q1 = np.array(
        [
            [dt**4 / 4.0, dt**3 / 2.0],
            [dt**3 / 2.0, dt**2],
        ],
        dtype=float,
    )
    q_block = q * q1
    q_mat = np.zeros((4, 4), dtype=float)
    q_mat[np.ix_([0, 2], [0, 2])] = q_block
    q_mat[np.ix_([1, 3], [1, 3])] = q_block

    h = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    return f, q_mat, h


def track_positions_cv2d(
    measured_xy: Iterable[Sequence[float]],
    cfg: TrackingConfig2D = TrackingConfig2D(),
    x0_xy: Optional[Sequence[float]] = None,
    v0_xy: Sequence[float] = (0.0, 0.0),
) -> List[TrackingPoint2D]:
    """
    Run a 2D constant-velocity Kalman tracker over position measurements.

    Input:
    - `measured_xy`: sequence of [x, y] measurements.
    - `cfg`: tracking noise/timing configuration.
    - `x0_xy`: optional initial position; default is first measurement.
    - `v0_xy`: initial velocity guess.
    """
    z_list = [np.asarray(z, dtype=float).reshape(2) for z in measured_xy]
    if not z_list:
        return []

    f, q_mat, h = _state_matrices(cfg.dt_s, cfg.process_accel_sigma_mps2)
    r_var = max(float(cfg.measurement_sigma_m), 1e-6) ** 2
    r = np.array([[r_var, 0.0], [0.0, r_var]], dtype=float)
    eye4 = np.eye(4, dtype=float)

    # Initial state uses first measurement position unless user overrides it.
    p0 = np.asarray(x0_xy, dtype=float).reshape(2) if x0_xy is not None else z_list[0]
    v0 = np.asarray(v0_xy, dtype=float).reshape(2)
    x = np.array([p0[0], p0[1], v0[0], v0[1]], dtype=float)

    p = np.diag(
        [
            float(cfg.init_pos_sigma_m) ** 2,
            float(cfg.init_pos_sigma_m) ** 2,
            float(cfg.init_vel_sigma_mps) ** 2,
            float(cfg.init_vel_sigma_mps) ** 2,
        ]
    ).astype(float)

    out: List[TrackingPoint2D] = []
    for k, z in enumerate(z_list):
        # Predict state and covariance to current frame.
        x_pred = f @ x
        p_pred = f @ p @ f.T + q_mat

        # Innovation (measurement residual) and Kalman gain.
        y = z - (h @ x_pred)
        s = h @ p_pred @ h.T + r
        k_gain = p_pred @ h.T @ np.linalg.inv(s)

        # Correct predicted state with measurement.
        x = x_pred + k_gain @ y
        p = (eye4 - k_gain @ h) @ p_pred

        out.append(
            TrackingPoint2D(
                step=int(k),
                measured_xy=(float(z[0]), float(z[1])),
                predicted_xy=(float(x_pred[0]), float(x_pred[1])),
                filtered_xy=(float(x[0]), float(x[1])),
                innovation_xy=(float(y[0]), float(y[1])),
            )
        )

    return out
