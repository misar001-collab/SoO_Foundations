#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3 Step 3 command-line evaluation runner.

This script:
1) runs Monte Carlo evaluation over noise levels and geometries
2) writes a CSV table of metrics
3) generates a report plot for presentations
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from soo_foundations.s3_evaluation import default_geometries_2d, evaluate_solver_monte_carlo


def parse_noise_list(noise_csv: str) -> list[float]:
    """
    Parse comma-separated noise values.

    Example input:
      "1,2,4,8,12"
    """
    vals = []
    for token in noise_csv.split(","):
        t = token.strip()
        if t:
            vals.append(float(t))
    if not vals:
        raise ValueError("noise list is empty")
    return vals


def write_results_csv(path: Path, rows: list[dict]) -> None:
    """Write aggregated Monte Carlo metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "geometry",
        "noise_sigma_m",
        "runs",
        "rmse_m",
        "median_err_m",
        "p95_err_m",
        "convergence_rate_pct",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_results(path: Path, rows: list[dict], title: str, dpi: int = 220) -> None:
    """
    Plot two panels:
    - RMSE vs noise for each geometry
    - Convergence rate vs noise for each geometry
    """
    # Group rows by geometry name for clean multi-line plots.
    by_geom: dict[str, list[dict]] = {}
    for r in rows:
        by_geom.setdefault(str(r["geometry"]), []).append(r)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel 1: RMSE trend.
    ax = axes[0]
    for geom, g_rows in by_geom.items():
        g_rows = sorted(g_rows, key=lambda x: float(x["noise_sigma_m"]))
        x = np.array([float(r["noise_sigma_m"]) for r in g_rows], dtype=float)
        y = np.array([float(r["rmse_m"]) for r in g_rows], dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.4, label=geom)
    ax.set_title("Stage 3 Monte Carlo: RMSE vs Noise")
    ax.set_xlabel("Range Noise Sigma (m)")
    ax.set_ylabel("Position RMSE (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Panel 2: convergence trend.
    ax = axes[1]
    for geom, g_rows in by_geom.items():
        g_rows = sorted(g_rows, key=lambda x: float(x["noise_sigma_m"]))
        x = np.array([float(r["noise_sigma_m"]) for r in g_rows], dtype=float)
        y = np.array([float(r["convergence_rate_pct"]) for r in g_rows], dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.4, label=geom)
    ax.set_title("Stage 3 Monte Carlo: Convergence vs Noise")
    ax.set_xlabel("Range Noise Sigma (m)")
    ax.set_ylabel("Convergence Rate (%)")
    ax.set_ylim(0, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi))
    print(f"Saved Stage 3 Monte Carlo plot: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise_sigmas_m", default="1,2,4,8,12,16", help="comma-separated list, e.g. 1,2,4,8")
    ap.add_argument("--runs_per_case", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max_iters", type=int, default=40)
    ap.add_argument("--x0_x", type=float, default=0.0)
    ap.add_argument("--x0_y", type=float, default=0.0)
    ap.add_argument("--out_csv", default="results/s3_monte_carlo_metrics.csv")
    ap.add_argument("--out_png", default="results/s3_monte_carlo_metrics.png")
    ap.add_argument("--title", default="Stage 3 Monte Carlo Evaluation")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--no_show", action="store_true")
    args = ap.parse_args()

    noise_sigmas = parse_noise_list(args.noise_sigmas_m)
    geoms = default_geometries_2d()

    rows = evaluate_solver_monte_carlo(
        geometries=geoms,
        noise_sigmas_m=noise_sigmas,
        runs_per_case=int(args.runs_per_case),
        seed=int(args.seed),
        max_iters=int(args.max_iters),
        x0_xy=(float(args.x0_x), float(args.x0_y)),
    )

    out_csv = Path(args.out_csv)
    write_results_csv(out_csv, rows)
    print(f"Saved Stage 3 Monte Carlo metrics CSV: {out_csv}")

    out_png = Path(args.out_png)
    plot_results(out_png, rows, title=args.title, dpi=int(args.dpi))

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

