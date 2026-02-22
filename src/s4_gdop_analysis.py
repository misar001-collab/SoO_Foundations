#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 4 GDOP analysis for multi-static ellipse intersection geometry.

This script builds intuition for Geometric Dilution of Precision (GDOP):
- It uses 3 transmitter towers and 1 receiver (multi-static setup).
- It moves the target toward the baseline (line between a selected tower and
  receiver) by sweeping perpendicular distance-to-baseline values.
- For each target position, it runs Monte Carlo noisy range measurements and
  solves target position with weighted least squares.
- It reports and plots how geometry-only GDOP and empirical error change.

Key intuition:
- As the target gets closer to a baseline, ellipse intersections become less
  well-conditioned, so estimate clouds become more "blurred" (higher spread).
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from soo_foundations.s3_solver import (
    make_bistatic_measurements_2d,
    predict_bistatic_range_2d,
    solve_position_bistatic_wls_2d,
)


def parse_xy(text: str) -> np.ndarray:
    """Parse one point from 'x,y' text."""
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected point as 'x,y'. Got: {text}")
    return np.array([float(parts[0]), float(parts[1])], dtype=float)


def parse_xy_list(text: str) -> np.ndarray:
    """Parse semicolon-separated points: 'x1,y1;x2,y2;x3,y3'."""
    pts = [parse_xy(item.strip()) for item in str(text).split(";") if item.strip()]
    if len(pts) < 3:
        raise ValueError("Need at least 3 transmitter points for multi-static GDOP analysis.")
    return np.vstack(pts).astype(float)


def parse_float_list(text: str) -> list[float]:
    """Parse comma-separated numeric values."""
    vals = [v.strip() for v in str(text).split(",") if v.strip()]
    if not vals:
        raise ValueError("At least one numeric value is required.")
    return [float(v) for v in vals]


def unit_vec(v: np.ndarray) -> np.ndarray:
    """Return normalized vector with tiny-floor safety."""
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Cannot normalize zero-length vector.")
    return v / n


def jacobian_row(target_xy: np.ndarray, tx_xy: np.ndarray, rx_xy: np.ndarray) -> np.ndarray:
    """
    One Jacobian row for bistatic range wrt [x, y].

    For r = |p-tx| + |p-rx|:
      dr/dp = (p-tx)/|p-tx| + (p-rx)/|p-rx|
    """
    p = np.asarray(target_xy, dtype=float)
    tx = np.asarray(tx_xy, dtype=float)
    rx = np.asarray(rx_xy, dtype=float)
    v1 = p - tx
    v2 = p - rx
    d1 = max(float(np.linalg.norm(v1)), 1e-9)
    d2 = max(float(np.linalg.norm(v2)), 1e-9)
    return (v1 / d1) + (v2 / d2)


def geometry_metrics(target_xy: np.ndarray, txs_xy: np.ndarray, rx_xy: np.ndarray) -> tuple[float, float]:
    """
    Compute geometry-only GDOP and conditioning from Jacobian.

    We form G = H^T H from Jacobian H. Then:
    - geometry_cov = inv(G)
    - gdop = sqrt(trace(geometry_cov))
    - cond_G = condition number of G
    """
    h = np.vstack([jacobian_row(target_xy, tx, rx_xy) for tx in txs_xy]).astype(float)
    g = h.T @ h
    # Small damping protects inversion in near-singular geometries.
    g_damped = g + 1e-10 * np.eye(2, dtype=float)
    cov_geom = np.linalg.inv(g_damped)
    gdop = float(np.sqrt(np.trace(cov_geom)))
    cond_g = float(np.linalg.cond(g_damped))
    return gdop, cond_g


def run_one_solve(
    rng: np.random.Generator,
    target_true_xy: np.ndarray,
    txs_xy: np.ndarray,
    rx_xy: np.ndarray,
    noise_sigma_m: float,
    max_iters: int,
    tol_step_m: float,
) -> tuple[bool, np.ndarray]:
    """Run one noisy multi-static least-squares solve."""
    measured_ranges = []
    for tx in txs_xy:
        r_true = predict_bistatic_range_2d(target_true_xy, tx, rx_xy)
        r_meas = r_true + rng.normal(0.0, float(noise_sigma_m))
        measured_ranges.append(float(r_meas))

    meas = make_bistatic_measurements_2d(
        measured_ranges_m=measured_ranges,
        tx_positions_xy=txs_xy,
        rx_positions_xy=[rx_xy for _ in range(len(txs_xy))],
        sigma_ranges_m=[float(noise_sigma_m)] * len(txs_xy),
        confidences=[1.0] * len(txs_xy),
    )

    out = solve_position_bistatic_wls_2d(
        measurements=meas,
        x0_xy=None,
        max_iters=int(max_iters),
        tol_step_m=float(tol_step_m),
    )
    return bool(out.success), np.array(out.estimated_xy, dtype=float)


def main() -> None:
    # CLI exposes geometry, baseline experiment sweep, and output controls.
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=17, help="random seed")
    ap.add_argument("--runs_per_case", type=int, default=300, help="Monte Carlo runs for each target distance")
    ap.add_argument("--noise_sigma_m", type=float, default=8.0, help="range noise std-dev per measurement")
    ap.add_argument(
        "--txs_xy",
        type=str,
        default="0,0;16000,2500;-12000,9500",
        help="tower points: x1,y1;x2,y2;x3,y3",
    )
    ap.add_argument("--rx_xy", type=str, default="3000,-2500", help="receiver point: x,y")
    ap.add_argument("--baseline_tx_index", type=int, default=1, help="1-based tx index that defines baseline with receiver")
    ap.add_argument(
        "--baseline_distances_m",
        type=str,
        default="6000,4000,2500,1500,800,400,200",
        help="comma-separated perpendicular distances to baseline",
    )
    ap.add_argument("--baseline_ratio", type=float, default=0.45, help="anchor ratio along selected tx->rx line [0,1]")
    ap.add_argument("--max_iters", type=int, default=50, help="WLS max iterations")
    ap.add_argument("--tol_step_m", type=float, default=1e-6, help="WLS step tolerance")
    ap.add_argument("--out_csv", type=str, default="results/s4_gdop_analysis.csv", help="output CSV path")
    ap.add_argument("--save_png", type=str, default="results/s4_gdop_analysis.png", help="output figure path")
    ap.add_argument("--no_show", action="store_true", help="disable interactive plot window")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    txs = parse_xy_list(args.txs_xy)
    rx = parse_xy(args.rx_xy)
    dists = sorted(parse_float_list(args.baseline_distances_m), reverse=True)

    # Select one tower to define the baseline used in the GDOP intuition sweep.
    idx = int(args.baseline_tx_index) - 1
    if idx < 0 or idx >= len(txs):
        raise SystemExit(f"--baseline_tx_index must be in [1, {len(txs)}].")
    tx_base = txs[idx]

    # Build baseline anchor and its perpendicular unit vector.
    line = rx - tx_base
    u_line = unit_vec(line)
    u_perp = np.array([-u_line[1], u_line[0]], dtype=float)
    anchor = tx_base + float(args.baseline_ratio) * line

    rows = []
    cloud_far = None
    cloud_near = None
    target_far = None
    target_near = None

    for dist in dists:
        # Target is placed at fixed along-line anchor plus adjustable perpendicular offset.
        target = anchor + float(dist) * u_perp

        gdop, cond_g = geometry_metrics(target, txs, rx)

        ests = []
        errs = []
        success_count = 0
        for _ in range(int(args.runs_per_case)):
            ok, est = run_one_solve(
                rng=rng,
                target_true_xy=target,
                txs_xy=txs,
                rx_xy=rx,
                noise_sigma_m=float(args.noise_sigma_m),
                max_iters=int(args.max_iters),
                tol_step_m=float(args.tol_step_m),
            )
            if ok and np.all(np.isfinite(est)):
                success_count += 1
                ests.append(est)
                errs.append(float(np.linalg.norm(est - target)))

        ests_arr = np.vstack(ests) if ests else np.zeros((0, 2), dtype=float)
        errs_arr = np.asarray(errs, dtype=float)

        rmse = float(np.sqrt(np.mean(errs_arr * errs_arr))) if errs_arr.size else np.inf
        med = float(np.median(errs_arr)) if errs_arr.size else np.inf
        p95 = float(np.percentile(errs_arr, 95.0)) if errs_arr.size else np.inf
        conv_pct = 100.0 * float(success_count) / float(max(int(args.runs_per_case), 1))

        # "Blur area" metric: area of 1-sigma estimate covariance ellipse.
        if ests_arr.shape[0] >= 3:
            cov = np.cov(ests_arr[:, 0], ests_arr[:, 1], ddof=1)
            blur_area = float(np.pi * np.sqrt(max(np.linalg.det(cov), 0.0)))
        else:
            blur_area = np.inf

        rows.append(
            {
                "distance_to_baseline_m": float(dist),
                "gdop": gdop,
                "cond_hth": cond_g,
                "convergence_rate_pct": conv_pct,
                "rmse_m": rmse,
                "median_err_m": med,
                "p95_err_m": p95,
                "blur_area_m2": blur_area,
                "target_x_m": float(target[0]),
                "target_y_m": float(target[1]),
            }
        )

        if dist == max(dists):
            cloud_far = ests_arr
            target_far = target.copy()
        if dist == min(dists):
            cloud_near = ests_arr
            target_near = target.copy()

    # Write machine-readable summary for reporting and regression comparisons.
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "distance_to_baseline_m",
        "gdop",
        "cond_hth",
        "convergence_rate_pct",
        "rmse_m",
        "median_err_m",
        "p95_err_m",
        "blur_area_m2",
        "target_x_m",
        "target_y_m",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote CSV: {out_csv}")

    # Prepare arrays for plotting trends.
    x_dist = np.array([r["distance_to_baseline_m"] for r in rows], dtype=float)
    y_gdop = np.array([r["gdop"] for r in rows], dtype=float)
    y_rmse = np.array([r["rmse_m"] for r in rows], dtype=float)
    y_blur = np.array([r["blur_area_m2"] for r in rows], dtype=float)

    fig, axs = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

    # Panel 1: geometry map and target sweep points.
    ax = axs[0, 0]
    for i, tx in enumerate(txs, start=1):
        ax.scatter([tx[0]], [tx[1]], marker="^", s=120)
        ax.text(tx[0], tx[1], f" Tx{i}", fontsize=9)
    ax.scatter([rx[0]], [rx[1]], marker="s", s=140, label="Receiver")
    ax.plot([tx_base[0], rx[0]], [tx_base[1], rx[1]], "k--", linewidth=1.5, label="Selected baseline")
    ax.scatter([r["target_x_m"] for r in rows], [r["target_y_m"] for r in rows], c=x_dist, cmap="viridis", s=65)
    ax.set_title("Geometry and Baseline-Distance Sweep")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel 2: GDOP, RMSE, and blur trend as target moves closer to baseline.
    ax = axs[0, 1]
    ax.plot(x_dist, y_gdop, "o-", label="GDOP")
    ax.plot(x_dist, y_rmse, "s-", label="RMSE (m)")
    ax.plot(x_dist, y_blur, "^-", label="Blur area (m^2)")
    ax.invert_xaxis()
    ax.set_title("Near-Baseline Degradation Trend")
    ax.set_xlabel("Distance to baseline (m) [decreasing -> worse geometry]")
    ax.set_ylabel("Metric value")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel 3: estimate cloud far from baseline (usually better conditioned).
    ax = axs[1, 0]
    if cloud_far is not None and cloud_far.size > 0:
        ax.scatter(cloud_far[:, 0], cloud_far[:, 1], s=10, alpha=0.25, label="Estimates")
    if target_far is not None:
        ax.scatter([target_far[0]], [target_far[1]], marker="x", s=160, c="red", label="True target")
    for i, tx in enumerate(txs, start=1):
        ax.scatter([tx[0]], [tx[1]], marker="^", s=90)
        ax.text(tx[0], tx[1], f" Tx{i}", fontsize=8)
    ax.scatter([rx[0]], [rx[1]], marker="s", s=110, label="Receiver")
    ax.set_title(f"Estimate Cloud (Far from baseline: {max(dists):.1f} m)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel 4: estimate cloud near baseline (usually more blurred).
    ax = axs[1, 1]
    if cloud_near is not None and cloud_near.size > 0:
        ax.scatter(cloud_near[:, 0], cloud_near[:, 1], s=10, alpha=0.25, label="Estimates")
    if target_near is not None:
        ax.scatter([target_near[0]], [target_near[1]], marker="x", s=160, c="red", label="True target")
    for i, tx in enumerate(txs, start=1):
        ax.scatter([tx[0]], [tx[1]], marker="^", s=90)
        ax.text(tx[0], tx[1], f" Tx{i}", fontsize=8)
    ax.scatter([rx[0]], [rx[1]], marker="s", s=110, label="Receiver")
    ax.set_title(f"Estimate Cloud (Near baseline: {min(dists):.1f} m)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_png = Path(args.save_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"Saved figure: {out_png}")

    # Print compact numeric summary so terminal-only runs remain useful.
    print("Summary by baseline distance:")
    for r in rows:
        print(
            f"  d={r['distance_to_baseline_m']:.1f} m | gdop={r['gdop']:.3f} | "
            f"rmse={r['rmse_m']:.3f} m | blur={r['blur_area_m2']:.1f} m^2 | "
            f"conv={r['convergence_rate_pct']:.1f}%"
        )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
