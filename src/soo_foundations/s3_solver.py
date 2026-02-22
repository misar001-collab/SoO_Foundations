#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3 - Step 2 baseline positioning solver (2D bistatic least squares).

Why this exists:
- Stage 2 gives us delay-derived bistatic range measurements.
- Stage 3 needs to turn those measurements into an estimated target position.

What this module does:
- Defines a 2D measurement structure (Tx, Rx, measured bistatic range).
- Uses weighted Gauss-Newton least squares to estimate target (x, y).
- Returns detailed solver diagnostics so behavior is easy to inspect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


EPS_DIST = 1e-9


@dataclass(frozen=True)
class BistaticMeasurement2D:
    """
    One bistatic range measurement in 2D.

    Interpretation:
      measured_range ~= |target - tx| + |target - rx|
    """

    tx_xy: Tuple[float, float]
    rx_xy: Tuple[float, float]
    measured_bistatic_range_m: float
    sigma_range_m: float = 1.0
    confidence: float = 1.0


@dataclass(frozen=True)
class SolverResult2D:
    """Output bundle from weighted least-squares solver."""

    success: bool
    iterations: int
    estimated_xy: Tuple[float, float]
    residual_rms_m: float
    weighted_cost: float
    reason: str


def _safe_norm(x: np.ndarray) -> float:
    """Euclidean norm with tiny floor to avoid divide-by-zero in Jacobian math."""
    return max(float(np.linalg.norm(x)), EPS_DIST)


def predict_bistatic_range_2d(target_xy: Sequence[float], tx_xy: Sequence[float], rx_xy: Sequence[float]) -> float:
    """
    Compute predicted bistatic range for one sensor pair.

    Geometry reminder:
    - Monostatic range is one path (radar -> target -> radar).
    - Bistatic range here is two different legs:
        transmitter -> target plus target -> receiver.
    """
    p = np.asarray(target_xy, dtype=float)
    tx = np.asarray(tx_xy, dtype=float)
    rx = np.asarray(rx_xy, dtype=float)
    d_tx = _safe_norm(p - tx)
    d_rx = _safe_norm(p - rx)
    return float(d_tx + d_rx)


def _jacobian_row_2d(target_xy: np.ndarray, tx_xy: np.ndarray, rx_xy: np.ndarray) -> np.ndarray:
    """
    Jacobian of bistatic range wrt target [x, y].

    For range r = |p-tx| + |p-rx|:
      dr/dp = (p-tx)/|p-tx| + (p-rx)/|p-rx|
    """
    v_tx = target_xy - tx_xy
    v_rx = target_xy - rx_xy
    d_tx = _safe_norm(v_tx)
    d_rx = _safe_norm(v_rx)
    return (v_tx / d_tx) + (v_rx / d_rx)


def _initial_guess(measurements: Sequence[BistaticMeasurement2D]) -> np.ndarray:
    """
    Build a practical starting guess from sensor geometry.

    We start at the centroid of all Tx/Rx points. This is simple and usually
    stable enough for Gauss-Newton to move toward the solution.
    """
    pts: List[np.ndarray] = []
    for m in measurements:
        pts.append(np.asarray(m.tx_xy, dtype=float))
        pts.append(np.asarray(m.rx_xy, dtype=float))
    arr = np.vstack(pts)
    return np.mean(arr, axis=0)


def solve_position_bistatic_wls_2d(
    measurements: Sequence[BistaticMeasurement2D],
    x0_xy: Optional[Sequence[float]] = None,
    max_iters: int = 30,
    tol_step_m: float = 1e-6,
    damping: float = 1e-4,
) -> SolverResult2D:
    """
    Weighted least-squares solve for target position in 2D.

    Notes for non-experts:
    - We cannot directly "invert" nonlinear range equations.
    - So we linearize around current guess, solve a small correction, and repeat.
    - This is standard Gauss-Newton optimization.
    """
    if len(measurements) < 3:
        return SolverResult2D(
            success=False,
            iterations=0,
            estimated_xy=(math.nan, math.nan),
            residual_rms_m=math.inf,
            weighted_cost=math.inf,
            reason="need_at_least_3_measurements",
        )

    p = np.asarray(x0_xy, dtype=float) if x0_xy is not None else _initial_guess(measurements)
    if p.shape != (2,):
        raise ValueError("x0_xy must contain exactly 2 values [x, y]")

    reason = "max_iters_reached"
    success = False
    last_r = np.zeros(len(measurements), dtype=float)
    last_w = np.ones(len(measurements), dtype=float)

    for it in range(1, int(max_iters) + 1):
        # Assemble residual vector r and Jacobian J for current guess p.
        r = np.zeros(len(measurements), dtype=float)
        j = np.zeros((len(measurements), 2), dtype=float)
        w = np.zeros(len(measurements), dtype=float)

        for i, m in enumerate(measurements):
            tx = np.asarray(m.tx_xy, dtype=float)
            rx = np.asarray(m.rx_xy, dtype=float)
            pred = predict_bistatic_range_2d(p, tx, rx)
            r[i] = float(m.measured_bistatic_range_m) - pred
            j[i, :] = _jacobian_row_2d(p, tx, rx)

            # Weight = confidence / variance.
            # Larger sigma means lower trust.
            sigma = max(float(m.sigma_range_m), 1e-6)
            conf = max(float(m.confidence), 0.0)
            w[i] = conf / (sigma * sigma)

        # Convert weights into diagonal matrix form for WLS normal equations.
        w_diag = np.diag(w)
        jt_w = j.T @ w_diag
        h = jt_w @ j + float(damping) * np.eye(2, dtype=float)
        g = jt_w @ r

        # Solve for incremental update step.
        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            reason = "singular_normal_matrix"
            break

        p = p + step
        step_norm = float(np.linalg.norm(step))
        last_r = r
        last_w = w

        if step_norm <= float(tol_step_m):
            success = True
            reason = "converged_step"
            break

    # Final metrics for reporting and quality checks.
    rms = float(np.sqrt(np.mean(last_r * last_r))) if last_r.size else math.inf
    weighted_cost = float(last_r.T @ np.diag(last_w) @ last_r) if last_r.size else math.inf
    return SolverResult2D(
        success=bool(success),
        iterations=int(it if "it" in locals() else 0),
        estimated_xy=(float(p[0]), float(p[1])),
        residual_rms_m=float(rms),
        weighted_cost=float(weighted_cost),
        reason=reason,
    )


def make_bistatic_measurements_2d(
    measured_ranges_m: Iterable[float],
    tx_positions_xy: Sequence[Sequence[float]],
    rx_positions_xy: Sequence[Sequence[float]],
    sigma_ranges_m: Optional[Iterable[float]] = None,
    confidences: Optional[Iterable[float]] = None,
) -> List[BistaticMeasurement2D]:
    """
    Convenience helper to build typed measurement objects from plain lists.

    All input arrays/lists must have equal length N (one per measurement).
    """
    ranges = list(float(x) for x in measured_ranges_m)
    txs = [tuple(float(v) for v in p) for p in tx_positions_xy]
    rxs = [tuple(float(v) for v in p) for p in rx_positions_xy]
    if not (len(ranges) == len(txs) == len(rxs)):
        raise ValueError("measured_ranges_m, tx_positions_xy, and rx_positions_xy must have same length")

    if sigma_ranges_m is None:
        sigmas = [1.0] * len(ranges)
    else:
        sigmas = [float(x) for x in sigma_ranges_m]
        if len(sigmas) != len(ranges):
            raise ValueError("sigma_ranges_m length must match number of measurements")

    if confidences is None:
        confs = [1.0] * len(ranges)
    else:
        confs = [float(x) for x in confidences]
        if len(confs) != len(ranges):
            raise ValueError("confidences length must match number of measurements")

    out: List[BistaticMeasurement2D] = []
    for i in range(len(ranges)):
        out.append(
            BistaticMeasurement2D(
                tx_xy=txs[i],
                rx_xy=rxs[i],
                measured_bistatic_range_m=ranges[i],
                sigma_range_m=sigmas[i],
                confidence=confs[i],
            )
        )
    return out

