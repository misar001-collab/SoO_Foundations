#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CAF heatmap plotting tool.

Computes a delay-Doppler CAF from input recordings and renders a presentation-
friendly heatmap around the strongest detection peak.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from soo_foundations.soo_caf import caf_fftbank, caf_peak


def load_c64(path: str, max_samples: int = 0) -> np.ndarray:
    """Load complex64 samples from a raw .c64 file with optional truncation."""
    x = np.fromfile(path, dtype=np.complex64)
    if max_samples and max_samples > 0:
        x = x[:max_samples]
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference/original .c64 path")
    ap.add_argument("--surv", required=True, help="surveillance/reflected .c64 path")
    ap.add_argument("--samp_rate", type=float, default=1e6, help="sample rate in Hz")
    ap.add_argument("--max_samples", type=int, default=200000, help="0=use all samples")
    ap.add_argument("--delay_min", type=int, default=0)
    ap.add_argument("--delay_max", type=int, default=5000)
    ap.add_argument("--delay_step", type=int, default=1)
    ap.add_argument("--fd_min", type=float, default=-80e3)
    ap.add_argument("--fd_max", type=float, default=80e3)
    ap.add_argument("--fd_bins", type=int, default=161)
    ap.add_argument("--nfft", type=int, default=0, help="0=auto power-of-two")
    ap.add_argument("--true_delay", type=int, default=None, help="optional truth delay")
    ap.add_argument("--true_doppler", type=float, default=None, help="optional truth Doppler")
    ap.add_argument("--show_contours", action="store_true", help="overlay magnitude contour lines")
    ap.add_argument("--clip_db", type=float, default=60.0, help="display range below peak (dB)")
    ap.add_argument("--save_png", type=str, default=None, help="optional output PNG path")
    args = ap.parse_args()

    s_ref = load_c64(args.ref, args.max_samples)
    s_surv = load_c64(args.surv, args.max_samples)
    if s_ref.size == 0 or s_surv.size == 0:
        raise SystemExit("No samples loaded; check --ref/--surv paths and file contents.")

    delays = np.arange(args.delay_min, args.delay_max + 1, args.delay_step, dtype=int)
    dopplers = np.linspace(args.fd_min, args.fd_max, args.fd_bins, dtype=float)

    caf = caf_fftbank(
        s_ref=s_ref,
        s_surv=s_surv,
        samp_rate=args.samp_rate,
        delays=delays,
        dopplers=dopplers,
        nfft=None if args.nfft == 0 else args.nfft,
    )
    pk_delay, pk_fd, _, pk_mag = caf_peak(caf, delays, dopplers)

    mag_db = 20.0 * np.log10(np.maximum(np.abs(caf), 1e-12))
    vmax = float(np.max(mag_db))
    vmin = vmax - float(args.clip_db)

    print("CAF peak estimate:")
    print(f"  delay_hat   = {pk_delay} samples")
    print(f"  doppler_hat = {pk_fd:.3f} Hz")
    print(f"  |CAF|       = {pk_mag:.6g}")

    plt.figure(figsize=(12, 6))
    extent = [delays[0], delays[-1], dopplers[0], dopplers[-1]]
    plt.imshow(
        mag_db,
        aspect="auto",
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    cbar = plt.colorbar()
    cbar.set_label("CAF Magnitude (dB)")

    if args.show_contours:
        levels = np.linspace(vmin, vmax, 7)
        dd, ff = np.meshgrid(delays, dopplers)
        plt.contour(dd, ff, mag_db, levels=levels, colors="white", linewidths=0.6, alpha=0.75)

    if args.true_delay is not None and args.true_doppler is not None:
        plt.scatter(
            [args.true_delay],
            [args.true_doppler],
            marker="o",
            s=180,
            facecolors="none",
            edgecolors="red",
            linewidths=2.0,
            label="Injected truth",
            zorder=3,
        )

    plt.scatter(
        [pk_delay],
        [pk_fd],
        marker="x",
        s=180,
        c="black",
        linewidths=2.2,
        label="Estimated peak",
        zorder=4,
    )

    plt.title("CAF Heatmap |CAF(delay, doppler)|")
    plt.xlabel("Delay (samples)")
    plt.ylabel("Doppler (Hz)")
    plt.legend(loc="best")
    plt.tight_layout()

    if args.save_png:
        out = Path(args.save_png)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300)
        print(f"Saved CAF heatmap to {out}")

    plt.show()


if __name__ == "__main__":
    main()
