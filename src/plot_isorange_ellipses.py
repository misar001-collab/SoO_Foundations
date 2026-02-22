#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Isorange ellipse visualizer for one transmitter/receiver pair.

Given one Tx/Rx pair and one or more bistatic range values (meters), this script
draws the corresponding isorange ellipses defined by:
|p - tx| + |p - rx| = bistatic_range
It can also compute the bistatic range from a provided target point.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from soo_foundations.s3_solver import predict_bistatic_range_2d


def parse_xy(text: str) -> np.ndarray:
    """Parse a comma-separated point string like '1000,200' into [x, y]."""
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError(f"Point must be 'x,y'. Got: {text}")
    return np.array([float(parts[0]), float(parts[1])], dtype=float)


def parse_float_list(text: str) -> list[float]:
    """Parse comma-separated float values into a Python list."""
    vals = [v.strip() for v in str(text).split(",") if v.strip() != ""]
    if not vals:
        raise ValueError("At least one value is required in comma-separated list.")
    return [float(v) for v in vals]


def ellipse_points_from_bistatic_range(
    tx_xy: np.ndarray,
    rx_xy: np.ndarray,
    bistatic_range_m: float,
    n_points: int = 720,
) -> np.ndarray:
    """
    Build 2D points for one isorange ellipse.

    Geometry:
    - Foci are Tx and Rx.
    - For ellipse: sum of distances to foci is constant (= bistatic range).
    - Valid only when bistatic_range_m >= focal_distance.
    """
    tx = np.asarray(tx_xy, dtype=float).reshape(2)
    rx = np.asarray(rx_xy, dtype=float).reshape(2)
    focal_dist = float(np.linalg.norm(rx - tx))
    r_sum = float(bistatic_range_m)

    if r_sum < focal_dist:
        raise ValueError(
            f"Invalid bistatic range {r_sum:.3f} m: must be >= Tx-Rx distance {focal_dist:.3f} m."
        )

    # Ellipse center is midpoint between foci.
    center = 0.5 * (tx + rx)

    # In canonical ellipse form:
    # 2a = constant sum of distances, and c = half focal distance.
    a = 0.5 * r_sum
    c = 0.5 * focal_dist
    b_sq = max(a * a - c * c, 0.0)
    b = float(np.sqrt(b_sq))

    # Local ellipse points before rotation/translation.
    t = np.linspace(0.0, 2.0 * np.pi, int(n_points), endpoint=True)
    x_local = a * np.cos(t)
    y_local = b * np.sin(t)
    p_local = np.column_stack((x_local, y_local))

    # Rotate local ellipse so major axis aligns with Tx->Rx direction.
    d = rx - tx
    theta = float(np.arctan2(d[1], d[0]))
    ct, st = np.cos(theta), np.sin(theta)
    rot = np.array([[ct, -st], [st, ct]], dtype=float)

    return p_local @ rot.T + center


def main() -> None:
    # CLI for plotting one or more isorange ellipses.
    ap = argparse.ArgumentParser()
    ap.add_argument("--tx_xy", required=True, help="transmitter x,y in meters (example: 0,0)")
    ap.add_argument("--rx_xy", required=True, help="receiver x,y in meters (example: 1000,0)")
    ap.add_argument(
        "--ranges_m",
        default=None,
        help="comma-separated bistatic ranges in meters (example: 1200,1400,1700)",
    )
    ap.add_argument(
        "--target_xy",
        default=None,
        help="optional target x,y. If --ranges_m is omitted, one range is derived from this target.",
    )
    ap.add_argument("--n_points", type=int, default=720, help="points per ellipse curve")
    ap.add_argument("--title", type=str, default="Isorange Ellipses")
    ap.add_argument("--save_png", type=str, default=None, help="optional output figure path")
    ap.add_argument("--no_show", action="store_true", help="disable interactive plot window")
    args = ap.parse_args()

    tx = parse_xy(args.tx_xy)
    rx = parse_xy(args.rx_xy)

    # Decide which ranges to draw:
    # - Explicit list from CLI, or
    # - One auto-derived range from an optional target point.
    if args.ranges_m is not None:
        ranges = parse_float_list(args.ranges_m)
    else:
        if args.target_xy is None:
            raise SystemExit("Provide --ranges_m or --target_xy.")
        target = parse_xy(args.target_xy)
        r = predict_bistatic_range_2d(target, tx, rx)
        ranges = [float(r)]
        print(f"Derived bistatic range from target {target.tolist()}: {r:.3f} m")

    # Remove exact duplicates to reduce redundant drawing.
    ranges = sorted(set(float(r) for r in ranges))

    plt.figure(figsize=(9, 8))

    # Draw each requested isorange contour.
    for r in ranges:
        pts = ellipse_points_from_bistatic_range(tx, rx, r, n_points=args.n_points)
        plt.plot(pts[:, 0], pts[:, 1], linewidth=2.0, label=f"R = {r:.1f} m")

    # Plot transmitter/receiver markers.
    plt.scatter([tx[0]], [tx[1]], s=120, marker="^", label="Tx")
    plt.scatter([rx[0]], [rx[1]], s=120, marker="s", label="Rx")

    # Optionally overlay target point used for derived range.
    if args.target_xy is not None:
        target = parse_xy(args.target_xy)
        plt.scatter([target[0]], [target[1]], s=140, marker="x", label="Target")

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
