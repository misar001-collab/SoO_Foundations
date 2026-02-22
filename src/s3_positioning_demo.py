#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3 Step 2 demo:
- Build synthetic bistatic measurements in 2D.
- Run weighted least-squares position estimation.
- Print understandable error metrics.
"""

import argparse

import numpy as np

from soo_foundations.s3_solver import (
    make_bistatic_measurements_2d,
    predict_bistatic_range_2d,
    solve_position_bistatic_wls_2d,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="random seed for repeatability")
    ap.add_argument("--noise_sigma_m", type=float, default=8.0, help="std-dev of range noise in meters")
    ap.add_argument("--x0_x", type=float, default=0.0, help="optional x start guess")
    ap.add_argument("--x0_y", type=float, default=0.0, help="optional y start guess")
    ap.add_argument("--max_iters", type=int, default=40)
    ap.add_argument("--tol_step_m", type=float, default=1e-6)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Known geometry for this demo:
    # - One illuminator/transmitter.
    # - Multiple receiver sites.
    # - One true target point to recover.
    tx = np.array([0.0, 0.0], dtype=float)
    rxs = np.array(
        [
            [15000.0, 0.0],
            [0.0, 15000.0],
            [-14000.0, 7000.0],
            [9000.0, -12000.0],
        ],
        dtype=float,
    )
    target_true = np.array([7000.0, 5000.0], dtype=float)

    # Build noisy measurements from truth so we can test solver quality.
    true_ranges = []
    meas_ranges = []
    sigmas = []
    confs = []
    for rx in rxs:
        r_true = predict_bistatic_range_2d(target_true, tx, rx)
        r_meas = r_true + rng.normal(0.0, float(args.noise_sigma_m))
        true_ranges.append(r_true)
        meas_ranges.append(r_meas)
        sigmas.append(float(args.noise_sigma_m))
        confs.append(1.0)

    # Convert raw lists into typed measurements used by solver.
    txs = [tx for _ in rxs]
    meas = make_bistatic_measurements_2d(
        measured_ranges_m=meas_ranges,
        tx_positions_xy=txs,
        rx_positions_xy=rxs,
        sigma_ranges_m=sigmas,
        confidences=confs,
    )

    # Solve from user-provided initial guess.
    out = solve_position_bistatic_wls_2d(
        measurements=meas,
        x0_xy=[args.x0_x, args.x0_y],
        max_iters=args.max_iters,
        tol_step_m=args.tol_step_m,
    )

    est = np.array(out.estimated_xy, dtype=float)
    err = float(np.linalg.norm(est - target_true))

    print("Stage 3 Positioning Demo")
    print(f"  success         = {out.success}")
    print(f"  reason          = {out.reason}")
    print(f"  iterations      = {out.iterations}")
    print(f"  target_true_xy  = ({target_true[0]:.3f}, {target_true[1]:.3f}) m")
    print(f"  target_est_xy   = ({est[0]:.3f}, {est[1]:.3f}) m")
    print(f"  position_err_m  = {err:.6f}")
    print(f"  residual_rms_m  = {out.residual_rms_m:.6f}")
    print(f"  weighted_cost   = {out.weighted_cost:.6f}")


if __name__ == "__main__":
    main()

