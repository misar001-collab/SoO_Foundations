#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2 CAF estimator and visualizer.

Loads reference/surveillance IQ recordings, computes a delay-Doppler CAF
surface, reports the strongest peak estimate, and optionally plots a heatmap.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from soo_foundations.soo_caf import caf_fftbank, caf_peak


def load_c64(path, max_samples=0):
    """Load raw interleaved complex64 samples from disk, with optional truncation."""
    x = np.fromfile(path, dtype=np.complex64)
    if max_samples and max_samples > 0:
        x = x[:max_samples]
    return x


def main():
    # CLI for computing and visualizing CAF over a configurable delay/Doppler grid.
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference/original .c64")
    ap.add_argument("--surv", required=True, help="surveillance/reflected .c64")
    ap.add_argument("--samp_rate", type=float, default=1e6)

    # Load-size control and delay grid definition.
    ap.add_argument("--max_samples", type=int, default=200000, help="0=all (careful)")
    ap.add_argument("--delay_min", type=int, default=0)
    ap.add_argument("--delay_max", type=int, default=5000)
    ap.add_argument("--delay_step", type=int, default=1)

    # Doppler grid definition.
    ap.add_argument("--fd_min", type=float, default=-80e3)
    ap.add_argument("--fd_max", type=float, default=80e3)
    ap.add_argument("--fd_bins", type=int, default=161)

    # Optional FFT override for CAF internals plus output image option.
    ap.add_argument("--nfft", type=int, default=0, help="0=auto power-of-two")
    ap.add_argument("--save_png", type=str, default=None, help="Optional path to save CAF heatmap PNG")

    # Optional injected-truth values used only for reporting/plot overlays.
    ap.add_argument("--true_delay", type=int, default=None, help="Injected delay (samples)")
    ap.add_argument("--true_doppler", type=float, default=None, help="Injected doppler (Hz)")

    args = ap.parse_args()

    # Load both channels from disk and truncate equally (via min length inside caf_fftbank).
    s_ref = load_c64(args.ref, args.max_samples)
    s_surv = load_c64(args.surv, args.max_samples)

    # Build search axes.
    delays = np.arange(args.delay_min, args.delay_max + 1, args.delay_step, dtype=int)
    dopplers = np.linspace(args.fd_min, args.fd_max, args.fd_bins, dtype=float)

    # Compute CAF matrix for all requested delay/Doppler hypotheses.
    caf = caf_fftbank(
        s_ref=s_ref,
        s_surv=s_surv,
        samp_rate=args.samp_rate,
        delays=delays,
        dopplers=dopplers,
        nfft=None if args.nfft == 0 else args.nfft,
    )

    # Extract global peak estimate as delay/Doppler solution.
    pk_delay, pk_fd, _, pk_mag = caf_peak(caf, delays, dopplers)

    print("CAF peak estimate:")
    print(f"  delay_hat   = {pk_delay} samples")
    print(f"  doppler_hat = {pk_fd:.3f} Hz")
    print(f"  |CAF|       = {pk_mag:.6g}")

    # If ground truth is provided, report estimation errors.
    if args.true_delay is not None and args.true_doppler is not None:
        d_delay = pk_delay - int(args.true_delay)
        d_fd = pk_fd - float(args.true_doppler)
        print("Injected ground truth:")
        print(f"  delay_true   = {int(args.true_delay)} samples")
        print(f"  doppler_true = {float(args.true_doppler):.3f} Hz")
        print("Errors:")
        print(f"  d_delay   = {d_delay} samples")
        print(f"  d_doppler = {d_fd:.3f} Hz")

    # Visualize CAF magnitude in dB for interpretability.
    mag_db = 20 * np.log10(np.maximum(np.abs(caf), 1e-12))

    plt.figure(figsize=(12, 6))
    extent = [delays[0], delays[-1], dopplers[0], dopplers[-1]]
    plt.imshow(mag_db, aspect="auto", origin="lower", extent=extent)
    plt.colorbar(label="CAF Magnitude (dB)")
    plt.title("Cross Ambiguity Function |CAF(delay, doppler)|")
    plt.xlabel("Delay (samples)")
    plt.ylabel("Doppler (Hz)")

    # Draw injected truth first so users can compare truth vs estimate visually.
    if args.true_delay is not None and args.true_doppler is not None:
        plt.scatter(
            [args.true_delay],
            [args.true_doppler],
            marker="o",
            s=220,
            facecolors="none",
            edgecolors="red",
            linewidths=3.0,
            label="Injected truth",
            zorder=3,
        )
        plt.text(args.true_delay, args.true_doppler, " truth", color="red", fontsize=10, zorder=5)

    # Draw estimated peak as a large black X over the heatmap.
    plt.scatter(
        [pk_delay],
        [pk_fd],
        marker="x",
        s=200,
        c="black",
        linewidths=3.0,
        label="Estimated peak",
        zorder=4,
    )
    plt.text(pk_delay, pk_fd, " peak", color="black", fontsize=10, zorder=6)

    plt.legend()
    plt.tight_layout()

    if args.save_png is not None:
        plt.savefig(args.save_png, dpi=300)
        print(f"Saved CAF heatmap to {args.save_png}")

    plt.show()


if __name__ == "__main__":
    main()
