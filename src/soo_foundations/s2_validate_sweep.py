#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from .soo_caf import caf_fftbank, caf_peak
from .soo_sim import generate_chirp_pair


def bin_width(x: np.ndarray) -> float:
    """Return uniform-grid spacing estimated from endpoints."""
    if x.size <= 1:
        return 0.0
    return float((x[-1] - x[0]) / (x.size - 1))


def in_grid(value: float, grid: np.ndarray) -> bool:
    """Inclusive bounds check for scalar membership in a sorted 1D grid."""
    if grid.size == 0:
        return False
    return float(grid[0]) <= float(value) <= float(grid[-1])


def peak_to_median_db(caf: np.ndarray) -> float:
    """
    Peak quality metric: 20*log10(max(|CAF|) / median(|CAF|)).

    Higher values indicate a cleaner, better-isolated ambiguity peak.
    """
    mag = np.abs(caf)
    peak = float(np.max(mag))
    med = float(np.median(mag))
    if med <= 0.0:
        return math.inf if peak > 0.0 else 0.0
    return 20.0 * math.log10(peak / med)


def main():
    # Sweep driver for Stage-2 delay/Doppler validation over multiple truth Dopplers.
    ap = argparse.ArgumentParser()

    # Signal config (chirp generally gives sharp CAF peaks for validation).
    ap.add_argument("--samp_rate", type=float, default=1e6)
    ap.add_argument("--duration", type=float, default=0.2, help="seconds (keep modest for sweeps)")
    ap.add_argument("--f_start", type=float, default=-100e3)
    ap.add_argument("--f_end", type=float, default=100e3)
    ap.add_argument("--amp", type=float, default=1.0)

    # Ground-truth reflection parameters held constant except Doppler (swept below).
    ap.add_argument("--true_delay", type=int, default=1200)
    ap.add_argument("--attn", type=float, default=0.9)
    ap.add_argument("--snr_db", type=float, default=10.0)

    # Truth Doppler sweep configuration.
    ap.add_argument("--fd_start", type=float, default=0.0)
    ap.add_argument("--fd_stop", type=float, default=80e3)
    ap.add_argument("--fd_cases", type=int, default=201, help="number of truth dopplers to test")

    # Global CAF search grid (used when center_on_truth is disabled).
    ap.add_argument("--delay_min", type=int, default=0)
    ap.add_argument("--delay_max", type=int, default=3000)
    ap.add_argument("--delay_step", type=int, default=1)
    ap.add_argument("--fd_min", type=float, default=0.0)
    ap.add_argument("--fd_max", type=float, default=80e3)
    ap.add_argument("--fd_bins", type=int, default=161)

    # Optional tracking-like mode: center CAF search grid around each injected truth.
    ap.add_argument(
        "--center_on_truth",
        action="store_true",
        help="Use a local CAF grid centered on each injected truth (tracking-style validation).",
    )
    ap.add_argument(
        "--delay_half_window",
        type=int,
        default=100,
        help="Half-width in samples for truth-centered delay search.",
    )
    ap.add_argument(
        "--fd_half_window",
        type=float,
        default=5000.0,
        help="Half-width in Hz for truth-centered Doppler search.",
    )

    # Basic pass/fail tolerances.
    ap.add_argument("--delay_tol", type=int, default=1, help="samples")
    ap.add_argument(
        "--doppler_tol_hz",
        type=float,
        default=None,
        help="If set, overrides auto tolerance logic and uses this fixed Hz tolerance.",
    )
    ap.add_argument(
        "--doppler_tol_mode",
        choices=("bin", "cpi", "max"),
        default="max",
        help="Auto tolerance source when --doppler_tol_hz is not set.",
    )
    ap.add_argument(
        "--doppler_cpi_k",
        type=float,
        default=0.5,
        help="CPI tolerance scale: tol_cpi = k / T_cpi where T_cpi = duration.",
    )

    # Edge guard settings (for avoiding unreliable edge-of-grid cases).
    ap.add_argument(
        "--edge_guard_mode",
        choices=("bin", "cpi", "max"),
        default="max",
        help="Doppler edge guard source. Cases with truth inside this guard are skipped from scoring.",
    )
    ap.add_argument(
        "--edge_guard_bins",
        type=int,
        default=2,
        help="Edge guard in Doppler bins: guard_bin_hz = edge_guard_bins * bin_width.",
    )
    ap.add_argument(
        "--edge_guard_cpi_k",
        type=float,
        default=1.0,
        help="CPI edge guard scale: guard_cpi_hz = k / T_cpi where T_cpi = duration.",
    )

    # Robust mode supplements basic error thresholds with quality/geometry checks.
    ap.add_argument(
        "--pass_mode",
        choices=("basic", "robust"),
        default="robust",
        help="basic: delay/fd error thresholds only; robust: adds grid, edge-peak, and peak-quality checks.",
    )
    ap.add_argument(
        "--min_peak_to_median_db",
        type=float,
        default=8.0,
        help="Robust mode threshold: minimum 20*log10(peak/median(|CAF|)).",
    )

    # Output and progress logging.
    ap.add_argument("--out_csv", type=str, default="results/stage2_sweep.csv")
    ap.add_argument("--progress_every", type=int, default=25, help="print progress every N cases (0 disables)")
    args = ap.parse_args()

    # Input validation for user-facing arguments.
    if args.fd_cases <= 0:
        raise ValueError("--fd_cases must be > 0")
    if args.fd_bins <= 1:
        raise ValueError("--fd_bins must be > 1")
    if args.delay_step <= 0:
        raise ValueError("--delay_step must be > 0")
    if args.duration <= 0.0:
        raise ValueError("--duration must be > 0")
    if args.samp_rate <= 0.0:
        raise ValueError("--samp_rate must be > 0")
    if args.delay_tol < 0:
        raise ValueError("--delay_tol must be >= 0")
    if args.doppler_cpi_k < 0.0:
        raise ValueError("--doppler_cpi_k must be >= 0")
    if args.edge_guard_bins < 0:
        raise ValueError("--edge_guard_bins must be >= 0")
    if args.edge_guard_cpi_k < 0.0:
        raise ValueError("--edge_guard_cpi_k must be >= 0")

    # Sweep truth Doppler values.
    true_fds = np.linspace(args.fd_start, args.fd_stop, args.fd_cases, dtype=float)

    # Pre-build global grids used when not operating in truth-centered mode.
    delays_global = np.arange(args.delay_min, args.delay_max + 1, args.delay_step, dtype=int)
    dopplers_global = np.linspace(args.fd_min, args.fd_max, args.fd_bins, dtype=float)
    doppler_bin_w_global = bin_width(dopplers_global)

    # Ensure output directory exists.
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    pass_count = 0
    scored_count = 0
    reason_counts = {}

    for idx, fd_true in enumerate(true_fds):
        if args.progress_every and (idx % args.progress_every) == 0:
            print(f"  progress: {idx}/{len(true_fds)} cases...")

        # Grid selection: fixed global grid or local grid centered on current truth.
        if args.center_on_truth:
            dmin = int(args.true_delay) - int(args.delay_half_window)
            dmax = int(args.true_delay) + int(args.delay_half_window)
            delays = np.arange(max(0, dmin), dmax + 1, args.delay_step, dtype=int)

            fmin = float(fd_true) - float(args.fd_half_window)
            fmax = float(fd_true) + float(args.fd_half_window)
            dopplers = np.linspace(fmin, fmax, args.fd_bins, dtype=float)
        else:
            delays = delays_global
            dopplers = dopplers_global

        # Per-case Doppler resolution and tolerance derivation.
        doppler_bin_w = bin_width(dopplers)
        tol_bin = 0.5 * doppler_bin_w
        tol_cpi = float(args.doppler_cpi_k) / float(args.duration)

        if args.doppler_tol_hz is not None:
            doppler_tol_case = float(args.doppler_tol_hz)
            doppler_tol_src = "user"
        else:
            if args.doppler_tol_mode == "bin":
                doppler_tol_case = tol_bin
                doppler_tol_src = "bin"
            elif args.doppler_tol_mode == "cpi":
                doppler_tol_case = tol_cpi
                doppler_tol_src = "cpi"
            else:
                doppler_tol_case = max(tol_bin, tol_cpi)
                doppler_tol_src = "max(bin,cpi)"

        # Edge guard: mark near-boundary Doppler cases as unscored if requested.
        guard_bin_hz = float(args.edge_guard_bins) * doppler_bin_w
        guard_cpi_hz = float(args.edge_guard_cpi_k) / float(args.duration)
        if args.edge_guard_mode == "bin":
            edge_guard_hz = guard_bin_hz
            edge_guard_src = "bin"
        elif args.edge_guard_mode == "cpi":
            edge_guard_hz = guard_cpi_hz
            edge_guard_src = "cpi"
        else:
            edge_guard_hz = max(guard_bin_hz, guard_cpi_hz)
            edge_guard_src = "max(bin,cpi)"

        # Generate one synthetic pair at this truth Doppler.
        s_ref, s_surv = generate_chirp_pair(
            samp_rate=args.samp_rate,
            duration=args.duration,
            f_start=args.f_start,
            f_end=args.f_end,
            amp=args.amp,
            delay_samps=args.true_delay,
            doppler_hz=float(fd_true),
            attn=args.attn,
            snr_db=args.snr_db,
        )

        # Compute CAF matrix and extract global peak estimate.
        caf = caf_fftbank(
            s_ref=s_ref,
            s_surv=s_surv,
            samp_rate=args.samp_rate,
            delays=delays,
            dopplers=dopplers,
            nfft=None,
        )

        pk_delay, pk_fd, _, pk_mag = caf_peak(caf, delays, dopplers)
        p2m_db = peak_to_median_db(caf)

        # Estimation errors relative to known injected truth.
        d_delay = int(pk_delay) - int(args.true_delay)
        d_fd = float(pk_fd) - float(fd_true)

        # Grid coverage checks for truth values.
        in_delay_grid = in_grid(float(args.true_delay), delays.astype(float))
        in_doppler_grid = in_grid(float(fd_true), dopplers.astype(float))
        truth_in_grid = in_delay_grid and in_doppler_grid

        # Detect truth/peak proximity to Doppler search boundaries.
        fd_lo = float(dopplers[0])
        fd_hi = float(dopplers[-1])
        truth_near_edge = ((float(fd_true) - fd_lo) < edge_guard_hz) or ((fd_hi - float(fd_true)) < edge_guard_hz)
        peak_near_edge = ((float(pk_fd) - fd_lo) < edge_guard_hz) or ((fd_hi - float(pk_fd)) < edge_guard_hz)

        # By policy, truth-near-edge cases are tracked but excluded from score totals.
        scored = not truth_near_edge

        # Explicit edge index check catches peaks exactly on grid borders.
        pk_delay_idx = int(np.argmin(np.abs(delays - int(pk_delay))))
        pk_fd_idx = int(np.argmin(np.abs(dopplers - float(pk_fd))))
        peak_on_edge = (
            pk_delay_idx == 0
            or pk_delay_idx == (delays.size - 1)
            or pk_fd_idx == 0
            or pk_fd_idx == (dopplers.size - 1)
        )

        # Basic pass criterion: absolute error thresholds only.
        pass_basic = (abs(d_delay) <= int(args.delay_tol)) and (abs(d_fd) <= doppler_tol_case)

        # Robust criterion additionally enforces quality and geometric sanity checks.
        pass_robust = (
            pass_basic
            and truth_in_grid
            and (not peak_on_edge)
            and (not peak_near_edge)
            and (p2m_db >= args.min_peak_to_median_db)
        )

        if scored:
            scored_count += 1
            ok = pass_robust if args.pass_mode == "robust" else pass_basic
            pass_count += int(ok)
        else:
            ok = False

        # Collect granular reason tags for post-run diagnostics.
        fail_reasons = []
        if truth_near_edge:
            fail_reasons.append("truth_near_edge")
        if not pass_basic:
            fail_reasons.append("error_tol")
        if not truth_in_grid:
            fail_reasons.append("truth_outside_grid")
        if peak_on_edge:
            fail_reasons.append("peak_on_edge")
        if peak_near_edge:
            fail_reasons.append("peak_near_edge")
        if p2m_db < args.min_peak_to_median_db:
            fail_reasons.append("weak_peak")
        if not fail_reasons:
            fail_reasons.append("none")

        fail_reason = "|".join(fail_reasons)
        reason_counts[fail_reason] = reason_counts.get(fail_reason, 0) + 1

        # Persist full per-case record for offline analysis.
        rows.append(
            [
                float(fd_true),
                float(pk_fd),
                float(d_fd),
                int(args.true_delay),
                int(pk_delay),
                int(d_delay),
                float(pk_mag),
                float(doppler_bin_w),
                float(tol_cpi),
                doppler_tol_src,
                float(doppler_tol_case),
                float(edge_guard_hz),
                edge_guard_src,
                int(truth_near_edge),
                int(peak_near_edge),
                int(scored),
                int(in_delay_grid),
                int(in_doppler_grid),
                int(peak_on_edge),
                float(p2m_db),
                int(pass_basic),
                int(pass_robust),
                fail_reason,
                int(ok),
            ]
        )

    # Emit CSV with a stable schema for downstream notebooks/scripts.
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "fd_true_hz",
                "fd_hat_hz",
                "fd_err_hz",
                "delay_true_samp",
                "delay_hat_samp",
                "delay_err_samp",
                "caf_peak_mag",
                "doppler_bin_width_hz",
                "doppler_cpi_tol_hz",
                "doppler_tol_source",
                "doppler_tol_hz",
                "doppler_edge_guard_hz",
                "doppler_edge_guard_source",
                "truth_fd_near_edge",
                "peak_fd_near_edge",
                "scored",
                "truth_delay_in_grid",
                "truth_fd_in_grid",
                "peak_on_edge",
                "peak_to_median_db",
                "pass_basic",
                "pass_robust",
                "fail_reason",
                "pass",
            ]
        )
        w.writerows(rows)

    total = len(true_fds)
    skipped = total - scored_count
    fail = scored_count - pass_count
    mode = "truth-centered local grid" if args.center_on_truth else "fixed global grid"

    print("Stage 2 Sweep Summary")
    print(f"  mode        = {mode}")
    print(f"  cases       = {total}")
    print(f"  scored      = {scored_count}")
    print(f"  skipped     = {skipped} (truth near Doppler edge guard)")
    print(f"  pass        = {pass_count}")
    print(f"  fail        = {fail} (on scored cases)")
    print(f"  pass_mode   = {args.pass_mode}")
    print(f"  delay_tol   = {int(args.delay_tol)} samples")

    if args.doppler_tol_hz is not None:
        print(f"  doppler_tol = {float(args.doppler_tol_hz):.3f} Hz (user specified)")
    else:
        tol_bin_global = 0.5 * doppler_bin_w_global
        tol_cpi_global = float(args.doppler_cpi_k) / float(args.duration)
        if args.doppler_tol_mode == "bin":
            print(f"  doppler_tol = {tol_bin_global:.3f} Hz (bin mode, 0.5 * bin_width={doppler_bin_w_global:.3f} Hz)")
        elif args.doppler_tol_mode == "cpi":
            print(
                f"  doppler_tol = {tol_cpi_global:.3f} Hz "
                f"(cpi mode, k/Tcpi with k={args.doppler_cpi_k:.3f}, Tcpi={args.duration:.6f} s)"
            )
        else:
            print(
                f"  doppler_tol = max(bin,cpi) = max({tol_bin_global:.3f}, {tol_cpi_global:.3f}) Hz "
                f"(k={args.doppler_cpi_k:.3f}, Tcpi={args.duration:.6f} s)"
            )

    guard_bin_global = float(args.edge_guard_bins) * doppler_bin_w_global
    guard_cpi_global = float(args.edge_guard_cpi_k) / float(args.duration)

    if args.edge_guard_mode == "bin":
        print(
            f"  edge_guard  = {guard_bin_global:.3f} Hz "
            f"(bin mode, bins={args.edge_guard_bins}, bin_width={doppler_bin_w_global:.3f} Hz)"
        )
    elif args.edge_guard_mode == "cpi":
        print(
            f"  edge_guard  = {guard_cpi_global:.3f} Hz "
            f"(cpi mode, k={args.edge_guard_cpi_k:.3f}, Tcpi={args.duration:.6f} s)"
        )
    else:
        print(
            f"  edge_guard  = max(bin,cpi) = max({guard_bin_global:.3f}, {guard_cpi_global:.3f}) Hz "
            f"(bins={args.edge_guard_bins}, k={args.edge_guard_cpi_k:.3f})"
        )

    if args.pass_mode == "robust":
        print(f"  min_p2m_db  = {float(args.min_peak_to_median_db):.2f} dB")

    top_reasons = sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
    if top_reasons:
        top_txt = ", ".join([f"{k}:{v}" for k, v in top_reasons])
        print(f"  reasons     = {top_txt}")

    print(f"  CSV saved   = {out_path}")


if __name__ == "__main__":
    main()
