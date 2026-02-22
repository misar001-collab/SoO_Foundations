#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 4 temporal tracking demo.

This script simulates a moving 2D target, adds noisy per-frame position
measurements (as if they came from Stage 3), and applies a constant-velocity
Kalman filter to smooth the trajectory over time.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from soo_foundations.s4_tracking import TrackingConfig2D, track_positions_cv2d


def main() -> None:
    # CLI parameters are chosen to support quick demo runs and easy tuning.
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=9, help="random seed for repeatability")
    ap.add_argument("--steps", type=int, default=80, help="number of time steps")
    ap.add_argument("--dt_s", type=float, default=0.5, help="time step in seconds")
    ap.add_argument("--noise_sigma_m", type=float, default=25.0, help="measurement noise std-dev (m)")
    ap.add_argument("--sim_accel_sigma_mps2", type=float, default=0.8, help="simulated target accel std-dev")
    ap.add_argument("--filter_accel_sigma_mps2", type=float, default=1.0, help="tracker process accel std-dev")
    ap.add_argument("--x0_true", type=float, default=2000.0, help="initial true x (m)")
    ap.add_argument("--y0_true", type=float, default=-1500.0, help="initial true y (m)")
    ap.add_argument("--vx_true", type=float, default=120.0, help="initial true vx (m/s)")
    ap.add_argument("--vy_true", type=float, default=75.0, help="initial true vy (m/s)")
    ap.add_argument("--save_png", type=str, default="results/s4_tracking_demo.png", help="output plot path")
    ap.add_argument("--no_show", action="store_true", help="disable interactive plot window")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Build synthetic ground-truth motion with random acceleration each step.
    dt = float(args.dt_s)
    n = int(args.steps)
    p = np.array([float(args.x0_true), float(args.y0_true)], dtype=float)
    v = np.array([float(args.vx_true), float(args.vy_true)], dtype=float)

    true_xy = np.zeros((n, 2), dtype=float)
    for k in range(n):
        a = rng.normal(0.0, float(args.sim_accel_sigma_mps2), size=2)
        p = p + v * dt + 0.5 * a * dt * dt
        v = v + a * dt
        true_xy[k, :] = p

    # Create noisy measurements that mimic frame-by-frame Stage 3 estimates.
    meas_xy = true_xy + rng.normal(0.0, float(args.noise_sigma_m), size=true_xy.shape)

    # Configure and run tracker.
    cfg = TrackingConfig2D(
        dt_s=dt,
        process_accel_sigma_mps2=float(args.filter_accel_sigma_mps2),
        measurement_sigma_m=float(args.noise_sigma_m),
        init_pos_sigma_m=max(3.0 * float(args.noise_sigma_m), 1.0),
        init_vel_sigma_mps=150.0,
    )
    track = track_positions_cv2d(measured_xy=meas_xy, cfg=cfg)
    filt_xy = np.array([tp.filtered_xy for tp in track], dtype=float)

    # Compute error summary to show tracking gain clearly.
    raw_err = np.linalg.norm(meas_xy - true_xy, axis=1)
    filt_err = np.linalg.norm(filt_xy - true_xy, axis=1)
    rmse_raw = float(np.sqrt(np.mean(raw_err * raw_err)))
    rmse_filt = float(np.sqrt(np.mean(filt_err * filt_err)))

    print("Stage 4 Tracking Demo")
    print(f"  steps             = {n}")
    print(f"  dt_s              = {dt:.3f}")
    print(f"  noise_sigma_m     = {float(args.noise_sigma_m):.3f}")
    print(f"  rmse_raw_m        = {rmse_raw:.6f}")
    print(f"  rmse_filtered_m   = {rmse_filt:.6f}")
    print(f"  rmse_improve_pct  = {(100.0 * (rmse_raw - rmse_filt) / max(rmse_raw, 1e-9)):.2f}")

    # Plot trajectory view and per-step error trend for presentation/debugging.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 11), constrained_layout=True)

    ax1.plot(true_xy[:, 0], true_xy[:, 1], "k-", linewidth=2.0, label="True trajectory")
    ax1.plot(meas_xy[:, 0], meas_xy[:, 1], ".", alpha=0.6, label="Noisy measurements")
    ax1.plot(filt_xy[:, 0], filt_xy[:, 1], "-", linewidth=2.0, label="Kalman filtered")
    ax1.set_title("Stage 4: 2D Temporal Tracking")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.axis("equal")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    t = np.arange(n, dtype=float) * dt
    ax2.plot(t, raw_err, label="Raw error")
    ax2.plot(t, filt_err, label="Filtered error")
    ax2.set_title("Position Error vs Time")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("error magnitude (m)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    out = Path(args.save_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved figure: {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
