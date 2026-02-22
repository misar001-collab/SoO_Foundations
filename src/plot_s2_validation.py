#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2 sweep validation plotter.

Reads Stage 2 sweep CSV results and visualizes detection/estimation behavior
to help validate delay and Doppler recovery performance.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(csv_path: Path):
    """Load Stage 2 sweep CSV rows as dictionaries."""
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in CSV: {csv_path}")
    return rows


def to_float(rows, key):
    return np.array([float(r[key]) for r in rows], dtype=float)


def to_int(rows, key):
    return np.array([int(r[key]) for r in rows], dtype=int)


def split_reason_counts(rows):
    """
    Count individual fail tags from the compound `fail_reason` field.

    Example row value: "error_tol|peak_on_edge" -> increments both tags.
    """
    counts = Counter()
    for r in rows:
        tags = [t.strip() for t in r["fail_reason"].split("|") if t.strip()]
        for tag in tags:
            counts[tag] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to Stage 2 sweep CSV")
    ap.add_argument(
        "--save_png",
        default="results/stage2_validation_report.png",
        help="Output image path for report figure",
    )
    ap.add_argument("--title", default="Stage 2 Validation Report")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--no_show", action="store_true", help="Save image only; do not open plot window")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = load_rows(csv_path)

    fd_true = to_float(rows, "fd_true_hz")
    fd_hat = to_float(rows, "fd_hat_hz")
    fd_err = to_float(rows, "fd_err_hz")
    delay_err = to_float(rows, "delay_err_samp")
    doppler_tol = to_float(rows, "doppler_tol_hz")
    p2m_db = to_float(rows, "peak_to_median_db")

    scored = to_int(rows, "scored") == 1
    passed = to_int(rows, "pass") == 1

    n_total = rows.__len__()
    n_scored = int(np.sum(scored))
    n_pass = int(np.sum(passed & scored))
    n_fail = n_scored - n_pass
    pass_rate = (100.0 * n_pass / n_scored) if n_scored > 0 else 0.0

    # Use scored subset for performance and correctness metrics.
    idx = scored
    x_true = fd_true[idx]
    x_hat = fd_hat[idx]
    x_err = fd_err[idx]
    y_delay_err = delay_err[idx]
    tol = doppler_tol[idx]
    p2m = p2m_db[idx]
    ok = passed[idx]

    # Sort by truth Doppler so tolerance/error traces are readable.
    ord_idx = np.argsort(x_true)
    x_true_s = x_true[ord_idx]
    x_err_s = x_err[ord_idx]
    tol_s = tol[ord_idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Estimated vs true Doppler scatter.
    ax = axes[0, 0]
    ax.scatter(x_true[~ok], x_hat[~ok], s=26, color="#c0392b", label="Fail")
    ax.scatter(x_true[ok], x_hat[ok], s=26, color="#1f77b4", label="Pass")
    if x_true.size > 0:
        lo = min(float(np.min(x_true)), float(np.min(x_hat)))
        hi = max(float(np.max(x_true)), float(np.max(x_hat)))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="Ideal y=x")
    ax.set_title("Doppler Estimate vs Truth (Scored Cases)")
    ax.set_xlabel("True Doppler (Hz)")
    ax.set_ylabel("Estimated Doppler (Hz)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Panel 2: Doppler error and tolerance envelopes.
    ax = axes[0, 1]
    ax.plot(x_true_s, x_err_s, color="#2c3e50", linewidth=1.3, label="fd error")
    ax.plot(x_true_s, tol_s, color="#27ae60", linestyle="--", linewidth=1.1, label="+tol")
    ax.plot(x_true_s, -tol_s, color="#27ae60", linestyle="--", linewidth=1.1, label="-tol")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Doppler Error vs Truth")
    ax.set_xlabel("True Doppler (Hz)")
    ax.set_ylabel("Error (Hz)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Panel 3: Delay error by truth Doppler.
    ax = axes[1, 0]
    ax.scatter(x_true[~ok], y_delay_err[~ok], s=26, color="#c0392b", label="Fail")
    ax.scatter(x_true[ok], y_delay_err[ok], s=26, color="#1f77b4", label="Pass")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Delay Error vs Truth Doppler")
    ax.set_xlabel("True Doppler (Hz)")
    ax.set_ylabel("Delay Error (samples)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Panel 4: Failure reason counts and summary metrics.
    ax = axes[1, 1]
    fail_rows = [r for i, r in enumerate(rows) if scored[i] and not passed[i]]
    reason_counts = split_reason_counts(fail_rows)

    if reason_counts:
        labels = list(reason_counts.keys())
        vals = np.array([reason_counts[k] for k in labels], dtype=int)
        order = np.argsort(vals)[::-1]
        labels = [labels[i] for i in order]
        vals = vals[order]
        ax.barh(labels, vals, color="#7f8c8d")
        ax.invert_yaxis()
        ax.set_title("Top Failure Reasons (Scored Fails)")
        ax.set_xlabel("Count")
        ax.grid(True, axis="x", alpha=0.3)
    else:
        ax.axis("off")
        ax.text(0.02, 0.80, "No scored failures.", fontsize=11)

    summary = (
        f"Cases: {n_total}\n"
        f"Scored: {n_scored}\n"
        f"Pass: {n_pass}\n"
        f"Fail: {n_fail}\n"
        f"Pass rate: {pass_rate:.1f}%\n"
        f"Median |fd err|: {np.median(np.abs(x_err)) if x_err.size else 0.0:.2f} Hz\n"
        f"Median |delay err|: {np.median(np.abs(y_delay_err)) if y_delay_err.size else 0.0:.2f} samples\n"
        f"Median peak/median: {np.median(p2m) if p2m.size else 0.0:.2f} dB"
    )
    fig.text(
        0.66,
        0.02,
        summary,
        fontsize=10,
        va="bottom",
        ha="left",
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "#f5f6fa", "edgecolor": "#dcdde1"},
    )

    fig.suptitle(args.title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.95])

    save_path = Path(args.save_png)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=args.dpi)
    print(f"Saved Stage 2 validation report: {save_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
