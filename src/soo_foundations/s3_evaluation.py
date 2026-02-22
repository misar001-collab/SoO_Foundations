#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3 - Step 3 simulation-driven evaluation utilities.

This module answers a practical question:
"How good is the Stage 3 solver under different noise and geometry settings?"

We do that by running many randomized trials (Monte Carlo), then aggregating:
- RMSE (root-mean-square position error)
- median error
- 95th percentile error
- convergence rate
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .s3_solver import make_bistatic_measurements_2d, predict_bistatic_range_2d, solve_position_bistatic_wls_2d


@dataclass(frozen=True)
class GeometryConfig2D:
    """Named geometry bundle for repeatable Monte Carlo experiments."""

    name: str
    tx_xy: Tuple[float, float]
    rx_positions_xy: Tuple[Tuple[float, float], ...]
    target_true_xy: Tuple[float, float]


def default_geometries_2d() -> List[GeometryConfig2D]:
    """
    Provide a small set of fixed geometries.

    Why multiple geometries:
    - Solver difficulty depends strongly on where sensors are.
    - One geometry can look excellent while another struggles.
    """
    return [
        GeometryConfig2D(
            name="balanced_cross",
            tx_xy=(0.0, 0.0),
            rx_positions_xy=((15000.0, 0.0), (0.0, 15000.0), (-14000.0, 7000.0), (9000.0, -12000.0)),
            target_true_xy=(7000.0, 5000.0),
        ),
        GeometryConfig2D(
            name="wide_aperture",
            tx_xy=(0.0, 0.0),
            rx_positions_xy=((22000.0, 6000.0), (-20000.0, 11000.0), (6000.0, -23000.0), (18000.0, -15000.0)),
            target_true_xy=(9000.0, 3000.0),
        ),
        GeometryConfig2D(
            name="biased_receivers",
            tx_xy=(0.0, 0.0),
            rx_positions_xy=((8000.0, 3000.0), (11000.0, -2000.0), (13000.0, 5000.0), (15000.0, 1000.0)),
            target_true_xy=(6000.0, 4500.0),
        ),
    ]


def _run_single_trial(
    rng: np.random.Generator,
    geom: GeometryConfig2D,
    noise_sigma_m: float,
    max_iters: int,
    x0_xy: Sequence[float],
) -> Tuple[float, bool]:
    """
    Run one noisy solve and return:
    - position error in meters
    - convergence flag
    """
    tx = np.array(geom.tx_xy, dtype=float)
    rxs = np.array(geom.rx_positions_xy, dtype=float)
    target_true = np.array(geom.target_true_xy, dtype=float)

    # Build noisy measured bistatic ranges from the known truth.
    measured_ranges = []
    for rx in rxs:
        r_true = predict_bistatic_range_2d(target_true, tx, rx)
        r_meas = float(r_true + rng.normal(0.0, float(noise_sigma_m)))
        measured_ranges.append(r_meas)

    measurements = make_bistatic_measurements_2d(
        measured_ranges_m=measured_ranges,
        tx_positions_xy=[tx for _ in rxs],
        rx_positions_xy=rxs,
        sigma_ranges_m=[float(noise_sigma_m)] * len(rxs),
        confidences=[1.0] * len(rxs),
    )
    out = solve_position_bistatic_wls_2d(
        measurements=measurements,
        x0_xy=list(float(v) for v in x0_xy),
        max_iters=int(max_iters),
    )
    est = np.array(out.estimated_xy, dtype=float)
    err = float(np.linalg.norm(est - target_true))
    return err, bool(out.success)


def evaluate_solver_monte_carlo(
    geometries: Iterable[GeometryConfig2D],
    noise_sigmas_m: Iterable[float],
    runs_per_case: int,
    seed: int = 0,
    max_iters: int = 40,
    x0_xy: Sequence[float] = (0.0, 0.0),
) -> List[Dict[str, float | int | str]]:
    """
    Evaluate solver over all (geometry x noise) combinations.

    Returned rows are plain dictionaries so they are easy to:
    - write to CSV
    - plot
    - inspect in notebooks
    """
    if runs_per_case <= 0:
        raise ValueError("runs_per_case must be > 0")

    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float | int | str]] = []

    for geom in geometries:
        for sigma in noise_sigmas_m:
            sigma = float(sigma)
            errors = np.zeros(runs_per_case, dtype=float)
            converged = np.zeros(runs_per_case, dtype=int)

            # Monte Carlo loop: same configuration, different random noise each run.
            for i in range(runs_per_case):
                err, ok = _run_single_trial(
                    rng=rng,
                    geom=geom,
                    noise_sigma_m=sigma,
                    max_iters=max_iters,
                    x0_xy=x0_xy,
                )
                errors[i] = err
                converged[i] = 1 if ok else 0

            rmse = float(np.sqrt(np.mean(errors * errors)))
            med = float(np.median(errors))
            p95 = float(np.percentile(errors, 95))
            conv_rate = float(100.0 * np.mean(converged))

            rows.append(
                {
                    "geometry": geom.name,
                    "noise_sigma_m": sigma,
                    "runs": int(runs_per_case),
                    "rmse_m": rmse,
                    "median_err_m": med,
                    "p95_err_m": p95,
                    "convergence_rate_pct": conv_rate,
                }
            )

    return rows

