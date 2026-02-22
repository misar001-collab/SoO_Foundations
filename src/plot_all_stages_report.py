#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified SoO report plotter for Stage 1 + Stage 2 + Stage 3.

This script is designed for presentation use:
- Stage 1 panels: show signal behavior in time/frequency.
- Stage 2 panels: show delay-Doppler estimation quality from sweep CSV.
- Stage 3 panels: show positioning solver behavior via Monte Carlo.

All panels are placed in one figure so you can quickly communicate full
pipeline progress without switching between multiple tools.
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


def load_c64(path: Path, max_samples: int | None = None) -> np.ndarray:
    """
    Load interleaved complex64 waveform from disk.

    We keep this helper local so this script has no hidden runtime dependencies.
    """
    x = np.fromfile(path, dtype=np.complex64)
    if max_samples is not None and max_samples > 0:
        x = x[:max_samples]
    return x


def mag_db(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert complex magnitude to dB with a floor to avoid log(0)."""
    return 20.0 * np.log10(np.maximum(np.abs(x), eps))


def fft_spectrum_db(x: np.ndarray, samp_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT-shifted spectrum for presentation plots.

    We apply a Hann window to reduce spectral leakage, so peaks are easier to
    compare visually.
    """
    n = int(len(x))
    if n == 0:
        return np.array([]), np.array([])
    w = np.hanning(n).astype(np.float64)
    xw = x.astype(np.complex128) * w
    x_fft = np.fft.fftshift(np.fft.fft(xw))
    f = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / float(samp_rate)))
    x_fft = x_fft / np.sum(w)
    return f, mag_db(x_fft)


def load_stage2_rows(csv_path: Path) -> list[dict[str, str]]:
    """Load Stage 2 sweep CSV rows as dictionaries."""
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def run_stage3_monte_carlo(
    runs: int,
    noise_sigma_m: float,
    seed: int,
    max_iters: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run repeated synthetic solves to summarize Stage 3 solver behavior.

    Returns:
    - `errors_m`: position error for each run
    - `residual_rms_m`: final residual RMS for each run
    - `converged`: 1 if solver converged, else 0
    - `est_xy`: estimated [x, y] for each run
    """
    rng = np.random.default_rng(seed)

    # Fixed geometry for fair apples-to-apples Monte Carlo comparison.
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

    errors_m = np.zeros(runs, dtype=float)
    residual_rms_m = np.zeros(runs, dtype=float)
    converged = np.zeros(runs, dtype=int)
    est_xy = np.zeros((runs, 2), dtype=float)

    for i in range(runs):
        measured_ranges = []
        for rx in rxs:
            r_true = predict_bistatic_range_2d(target_true, tx, rx)
            r_meas = float(r_true + rng.normal(0.0, noise_sigma_m))
            measured_ranges.append(r_meas)

        measurements = make_bistatic_measurements_2d(
            measured_ranges_m=measured_ranges,
            tx_positions_xy=[tx for _ in rxs],
            rx_positions_xy=rxs,
            sigma_ranges_m=[noise_sigma_m] * len(rxs),
            confidences=[1.0] * len(rxs),
        )
        out = solve_position_bistatic_wls_2d(
            measurements=measurements,
            x0_xy=[0.0, 0.0],
            max_iters=max_iters,
        )
        est = np.array(out.estimated_xy, dtype=float)
        err = float(np.linalg.norm(est - target_true))

        errors_m[i] = err
        residual_rms_m[i] = float(out.residual_rms_m)
        converged[i] = 1 if out.success else 0
        est_xy[i, :] = est

    return errors_m, residual_rms_m, converged, est_xy


def panel_message(ax: plt.Axes, title: str, message: str) -> None:
    """Show a clean message when a panel's data is unavailable."""
    ax.set_title(title)
    ax.axis("off")
    ax.text(0.02, 0.85, message, transform=ax.transAxes, fontsize=10, va="top")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_c64", type=str, default="data/original_chirp.c64")
    ap.add_argument("--reflected_c64", type=str, default="data/reflected_chirp.c64")
    ap.add_argument("--stage2_csv", type=str, default="results/stage2_sweep_present.csv")
    ap.add_argument("--samp_rate", type=float, default=1e6)
    ap.add_argument("--stage1_samples", type=int, default=120000, help="max samples for Stage 1 waveform load")
    ap.add_argument("--time_samples", type=int, default=5000, help="number of samples shown in time panel")
    ap.add_argument("--fft_len", type=int, default=65536, help="FFT length for Stage 1 spectrum panel")
    ap.add_argument("--mc_runs", type=int, default=200, help="Stage 3 Monte Carlo run count")
    ap.add_argument("--mc_noise_sigma_m", type=float, default=8.0, help="Stage 3 range-noise sigma in meters")
    ap.add_argument("--mc_seed", type=int, default=7, help="Stage 3 Monte Carlo random seed")
    ap.add_argument("--mc_max_iters", type=int, default=40)
    ap.add_argument("--title", type=str, default="SoO Foundations: Stage 1-3 Unified Report")
    ap.add_argument("--save_png", type=str, default="results/all_stages_report.png")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--no_show", action="store_true")
    args = ap.parse_args()

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # ------------------------------------------------------------------
    # Stage 1 panels
    # ------------------------------------------------------------------
    p_orig = Path(args.original_c64)
    p_refl = Path(args.reflected_c64)
    if p_orig.exists() and p_refl.exists():
        orig = load_c64(p_orig, max_samples=args.stage1_samples)
        refl = load_c64(p_refl, max_samples=args.stage1_samples)
        n = min(len(orig), len(refl))
        orig = orig[:n]
        refl = refl[:n]

        if n > 0:
            # Panel 1: time-domain magnitude overlay.
            n_time = min(int(args.time_samples), n)
            t = np.arange(n_time, dtype=float) / float(args.samp_rate)
            ax = axes[0, 0]
            ax.plot(t, np.abs(orig[:n_time]), label="Original", linewidth=1.1)
            ax.plot(t, np.abs(refl[:n_time]), label="Reflected", linewidth=1.1, alpha=0.9)
            ax.set_title("Stage 1: Signal Magnitude vs Time")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Magnitude")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

            # Panel 2: spectrum overlay.
            n_fft = min(int(args.fft_len), n)
            f_o, x_o = fft_spectrum_db(orig[:n_fft], args.samp_rate)
            f_r, x_r = fft_spectrum_db(refl[:n_fft], args.samp_rate)
            ax = axes[0, 1]
            ax.plot(f_o, x_o, label="Original", linewidth=1.1)
            ax.plot(f_r, x_r, label="Reflected", linewidth=1.1, alpha=0.9)
            ax.set_title("Stage 1: Spectrum (FFT Magnitude dB)")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude (dB)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
        else:
            panel_message(axes[0, 0], "Stage 1: Signal Magnitude vs Time", "Loaded files are empty.")
            panel_message(axes[0, 1], "Stage 1: Spectrum (FFT Magnitude dB)", "Loaded files are empty.")
    else:
        panel_message(
            axes[0, 0],
            "Stage 1: Signal Magnitude vs Time",
            f"Missing file(s):\n- {p_orig}\n- {p_refl}",
        )
        panel_message(
            axes[0, 1],
            "Stage 1: Spectrum (FFT Magnitude dB)",
            f"Missing file(s):\n- {p_orig}\n- {p_refl}",
        )

    # ------------------------------------------------------------------
    # Stage 2 panels
    # ------------------------------------------------------------------
    p_s2 = Path(args.stage2_csv)
    if p_s2.exists():
        rows = load_stage2_rows(p_s2)
        if rows:
            fd_true = np.array([float(r["fd_true_hz"]) for r in rows], dtype=float)
            fd_hat = np.array([float(r["fd_hat_hz"]) for r in rows], dtype=float)
            fd_err = np.array([float(r["fd_err_hz"]) for r in rows], dtype=float)
            tol = np.array([float(r.get("doppler_tol_hz", 0.0)) for r in rows], dtype=float)
            passed = np.array([int(r.get("pass", "0")) == 1 for r in rows], dtype=bool)
            scored = np.array([int(r.get("scored", "1")) == 1 for r in rows], dtype=bool)

            # Panel 3: estimate vs truth.
            ax = axes[1, 0]
            ok = passed & scored
            bad = (~passed) & scored
            ax.scatter(fd_true[ok], fd_hat[ok], s=20, label="Pass", color="#1f77b4")
            ax.scatter(fd_true[bad], fd_hat[bad], s=20, label="Fail", color="#c0392b")
            if np.any(scored):
                lo = float(min(np.min(fd_true[scored]), np.min(fd_hat[scored])))
                hi = float(max(np.max(fd_true[scored]), np.max(fd_hat[scored])))
                ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label="Ideal y=x")
            ax.set_title("Stage 2: Doppler Estimate vs Truth")
            ax.set_xlabel("True Doppler (Hz)")
            ax.set_ylabel("Estimated Doppler (Hz)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

            # Panel 4: error with tolerance envelope.
            ax = axes[1, 1]
            order = np.argsort(fd_true)
            x = fd_true[order]
            y = fd_err[order]
            t = tol[order]
            ax.plot(x, y, linewidth=1.1, color="#2c3e50", label="fd error")
            ax.plot(x, t, "--", linewidth=1.0, color="#27ae60", label="+tol")
            ax.plot(x, -t, "--", linewidth=1.0, color="#27ae60", label="-tol")
            ax.axhline(0.0, color="black", linewidth=1.0)
            ax.set_title("Stage 2: Doppler Error vs Truth")
            ax.set_xlabel("True Doppler (Hz)")
            ax.set_ylabel("Error (Hz)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
        else:
            panel_message(axes[1, 0], "Stage 2: Doppler Estimate vs Truth", "CSV exists but has no data rows.")
            panel_message(axes[1, 1], "Stage 2: Doppler Error vs Truth", "CSV exists but has no data rows.")
    else:
        panel_message(axes[1, 0], "Stage 2: Doppler Estimate vs Truth", f"Missing CSV:\n{p_s2}")
        panel_message(axes[1, 1], "Stage 2: Doppler Error vs Truth", f"Missing CSV:\n{p_s2}")

    # ------------------------------------------------------------------
    # Stage 3 panels (always computable because we simulate internally)
    # ------------------------------------------------------------------
    errs, rrms, conv, est_xy = run_stage3_monte_carlo(
        runs=int(args.mc_runs),
        noise_sigma_m=float(args.mc_noise_sigma_m),
        seed=int(args.mc_seed),
        max_iters=int(args.mc_max_iters),
    )
    conv_rate = 100.0 * float(np.mean(conv))

    # Panel 5: position error histogram.
    ax = axes[2, 0]
    ax.hist(errs, bins=24, color="#34495e", alpha=0.88, edgecolor="white")
    ax.axvline(np.median(errs), color="#f39c12", linestyle="--", linewidth=1.3, label="Median error")
    ax.axvline(np.percentile(errs, 95), color="#c0392b", linestyle="--", linewidth=1.3, label="95th pct")
    ax.set_title("Stage 3: Position Error Distribution")
    ax.set_xlabel("Position Error (m)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    # Panel 6: estimated position cloud + summary text.
    ax = axes[2, 1]
    target_true = np.array([7000.0, 5000.0], dtype=float)
    ax.scatter(est_xy[:, 0], est_xy[:, 1], s=14, alpha=0.5, label="Estimates")
    ax.scatter([target_true[0]], [target_true[1]], marker="x", s=120, linewidths=2.5, c="red", label="True target")
    ax.set_title("Stage 3: Estimated Position Cloud")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    summary = (
        f"runs={len(errs)}\n"
        f"convergence={conv_rate:.1f}%\n"
        f"median err={np.median(errs):.2f} m\n"
        f"95th err={np.percentile(errs, 95):.2f} m\n"
        f"median residual RMS={np.median(rrms):.2f} m"
    )
    ax.text(
        0.03,
        0.97,
        summary,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"},
    )

    fig.suptitle(args.title, fontsize=17, fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.96])

    out = Path(args.save_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(args.dpi))
    print(f"Saved unified report figure: {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

