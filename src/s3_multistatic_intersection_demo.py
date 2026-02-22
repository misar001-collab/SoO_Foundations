#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3 multi-static intersection demo (3 FM towers + 1 receiver).

This script simulates a target observed through three bistatic measurements,
where each measurement comes from one transmitter tower and one common receiver.
Each measurement defines one isorange ellipse in 2D. The script then uses the
existing weighted least-squares solver to estimate the target (x, y) where the
three ellipses intersect best under measurement noise.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from soo_foundations.s3_solver import (
    make_bistatic_measurements_2d,
    predict_bistatic_range_2d,
    solve_position_bistatic_wls_2d,
)


def parse_xy(text: str) -> np.ndarray:
    """Parse one point from 'x,y' text into a 2D NumPy vector."""
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected point format 'x,y'. Got: {text}")
    return np.array([float(parts[0]), float(parts[1])], dtype=float)


def parse_xy_list(text: str) -> np.ndarray:
    """Parse semicolon-separated points like 'x1,y1;x2,y2;x3,y3'."""
    items = [s.strip() for s in str(text).split(";") if s.strip()]
    pts = [parse_xy(item) for item in items]
    if len(pts) < 3:
        raise ValueError("Need at least 3 transmitter points for multi-static intersection.")
    return np.vstack(pts).astype(float)


def ellipse_points(tx_xy: np.ndarray, rx_xy: np.ndarray, bistatic_range_m: float, n_points: int) -> np.ndarray:
    """
    Generate one isorange ellipse curve from a Tx/Rx focus pair and range sum.

    Geometry summary for newcomers:
    - An ellipse is the set of points where distance to focus A plus distance to
      focus B is a constant.
    - Here, focus A is a transmitter tower and focus B is the receiver.
    """
    tx = np.asarray(tx_xy, dtype=float).reshape(2)
    rx = np.asarray(rx_xy, dtype=float).reshape(2)
    range_sum = float(bistatic_range_m)

    # Distance between foci must be <= the constant distance sum for a valid ellipse.
    focal_distance = float(np.linalg.norm(rx - tx))
    if range_sum < focal_distance:
        raise ValueError(
            f"Invalid ellipse: range {range_sum:.3f} m is smaller than Tx-Rx distance {focal_distance:.3f} m."
        )

    # Midpoint between Tx and Rx is the ellipse center in local geometry.
    center = 0.5 * (tx + rx)

    # Canonical ellipse parameters:
    # - 2a = range_sum (major axis length)
    # - c = half focus spacing
    # - b^2 = a^2 - c^2
    a = 0.5 * range_sum
    c = 0.5 * focal_distance
    b = float(np.sqrt(max(a * a - c * c, 0.0)))

    # Parametric points in local axis-aligned frame.
    t = np.linspace(0.0, 2.0 * np.pi, int(n_points), endpoint=True)
    p_local = np.column_stack((a * np.cos(t), b * np.sin(t)))

    # Rotate local ellipse to align major axis with Tx->Rx direction.
    d = rx - tx
    theta = float(np.arctan2(d[1], d[0]))
    ct, st = np.cos(theta), np.sin(theta)
    rot = np.array([[ct, -st], [st, ct]], dtype=float)

    return p_local @ rot.T + center


def main() -> None:
    # CLI arguments chosen so users can run a default demo immediately, then customize geometry.
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7, help="random seed for repeatable noise")
    ap.add_argument("--noise_sigma_m", type=float, default=8.0, help="std-dev of range noise in meters")
    ap.add_argument(
        "--txs_xy",
        type=str,
        default="0,0;16000,2500;-12000,9500",
        help="semicolon-separated transmitter points: x1,y1;x2,y2;x3,y3",
    )
    ap.add_argument("--rx_xy", type=str, default="3000,-2500", help="receiver point: x,y")
    ap.add_argument("--target_xy", type=str, default="7000,5200", help="true target point: x,y")
    ap.add_argument("--x0_xy", type=str, default=None, help="optional solver initial guess: x,y")
    ap.add_argument("--max_iters", type=int, default=50, help="solver max iterations")
    ap.add_argument("--tol_step_m", type=float, default=1e-6, help="solver step convergence tolerance")
    ap.add_argument("--n_points", type=int, default=720, help="points per plotted ellipse")
    ap.add_argument("--title", type=str, default="Stage 3 Multi-Static Ellipse Intersection")
    ap.add_argument("--save_png", type=str, default=None, help="optional output image path")
    ap.add_argument("--no_show", action="store_true", help="disable interactive figure window")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    txs = parse_xy_list(args.txs_xy)
    rx = parse_xy(args.rx_xy)
    target_true = parse_xy(args.target_xy)
    x0 = parse_xy(args.x0_xy) if args.x0_xy else None

    # Build one noisy bistatic range measurement for each transmitter with common receiver.
    true_ranges = []
    measured_ranges = []
    sigmas = []
    confidences = []
    for tx in txs:
        r_true = predict_bistatic_range_2d(target_true, tx, rx)
        r_meas = r_true + rng.normal(0.0, float(args.noise_sigma_m))
        true_ranges.append(float(r_true))
        measured_ranges.append(float(r_meas))
        sigmas.append(float(args.noise_sigma_m))
        confidences.append(1.0)

    # Convert plain arrays into typed measurements expected by the WLS solver.
    meas = make_bistatic_measurements_2d(
        measured_ranges_m=measured_ranges,
        tx_positions_xy=txs,
        rx_positions_xy=[rx for _ in range(len(txs))],
        sigma_ranges_m=sigmas,
        confidences=confidences,
    )

    # Solve for the target point that best fits all ellipse constraints together.
    out = solve_position_bistatic_wls_2d(
        measurements=meas,
        x0_xy=x0,
        max_iters=args.max_iters,
        tol_step_m=args.tol_step_m,
    )

    est = np.array(out.estimated_xy, dtype=float)
    pos_err_m = float(np.linalg.norm(est - target_true))

    # Console summary helps quick validation in non-plot workflows.
    print("Stage 3 Multi-Static Intersection Demo")
    print(f"  num_towers        = {len(txs)}")
    print(f"  success           = {out.success}")
    print(f"  reason            = {out.reason}")
    print(f"  iterations        = {out.iterations}")
    print(f"  target_true_xy_m  = ({target_true[0]:.3f}, {target_true[1]:.3f})")
    print(f"  target_est_xy_m   = ({est[0]:.3f}, {est[1]:.3f})")
    print(f"  position_err_m    = {pos_err_m:.6f}")
    print(f"  residual_rms_m    = {out.residual_rms_m:.6f}")
    print(f"  weighted_cost     = {out.weighted_cost:.6f}")
    for i, (r_t, r_m) in enumerate(zip(true_ranges, measured_ranges), start=1):
        print(f"  tower_{i}_range_true_m={r_t:.3f} range_meas_m={r_m:.3f}")

    # Plot all measured isorange ellipses and key geometry points.
    plt.figure(figsize=(10, 9))
    for i, (tx, r_m) in enumerate(zip(txs, measured_ranges), start=1):
        pts = ellipse_points(tx, rx, r_m, n_points=args.n_points)
        plt.plot(pts[:, 0], pts[:, 1], linewidth=2.0, label=f"Tower {i} ellipse")
        plt.scatter([tx[0]], [tx[1]], s=110, marker="^")
        plt.text(tx[0], tx[1], f" Tx{i}", fontsize=9)

    plt.scatter([rx[0]], [rx[1]], s=130, marker="s", label="Receiver")
    plt.text(rx[0], rx[1], " Rx", fontsize=9)

    plt.scatter([target_true[0]], [target_true[1]], s=140, marker="x", label="Target true")
    plt.scatter([est[0]], [est[1]], s=140, marker="o", facecolors="none", edgecolors="black", label="Target est")

    # Draw a visual error segment between truth and estimate.
    plt.plot([target_true[0], est[0]], [target_true[1], est[1]], "k--", linewidth=1.5, alpha=0.8)

    plt.title(args.title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.save_png:
        plt.savefig(args.save_png, dpi=150)
        print(f"Saved figure: {args.save_png}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
