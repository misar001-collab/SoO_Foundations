#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt

from soo_caf import caf_fftbank, caf_peak


def load_c64(path, max_samples=0):
    x = np.fromfile(path, dtype=np.complex64)
    if max_samples and max_samples > 0:
        x = x[:max_samples]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference/original .c64")
    ap.add_argument("--surv", required=True, help="surveillance/reflected .c64")
    ap.add_argument("--samp_rate", type=float, default=1e6)

    ap.add_argument("--max_samples", type=int, default=200000, help="0=all (careful)")
    ap.add_argument("--delay_min", type=int, default=0)
    ap.add_argument("--delay_max", type=int, default=5000)
    ap.add_argument("--delay_step", type=int, default=1)

    ap.add_argument("--fd_min", type=float, default=-80e3)
    ap.add_argument("--fd_max", type=float, default=80e3)
    ap.add_argument("--fd_bins", type=int, default=161)

    ap.add_argument("--nfft", type=int, default=0, help="0=auto power-of-two")
    ap.add_argument("--save_png", type=str, default=None, help="Optional path to save CAF heatmap PNG")

    # Validation inputs (optional)
    ap.add_argument("--true_delay", type=int, default=None, help="Injected delay (samples)")
    ap.add_argument("--true_doppler", type=float, default=None, help="Injected doppler (Hz)")

    args = ap.parse_args()

    s_ref = load_c64(args.ref, args.max_samples)
    s_surv = load_c64(args.surv, args.max_samples)

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

    pk_delay, pk_fd, pk_val, pk_mag = caf_peak(caf, delays, dopplers)

    # Print validation info
    print("CAF peak estimate:")
    print(f"  delay_hat   = {pk_delay} samples")
    print(f"  doppler_hat = {pk_fd:.3f} Hz")
    print(f"  |CAF|       = {pk_mag:.6g}")

    if args.true_delay is not None and args.true_doppler is not None:
        d_delay = pk_delay - int(args.true_delay)
        d_fd = pk_fd - float(args.true_doppler)
        print("Injected ground truth:")
        print(f"  delay_true   = {int(args.true_delay)} samples")
        print(f"  doppler_true = {float(args.true_doppler):.3f} Hz")
        print("Errors:")
        print(f"  Δdelay   = {d_delay} samples")
        print(f"  Δdoppler = {d_fd:.3f} Hz")

    # Plot heatmap of magnitude (dB)
    mag_db = 20 * np.log10(np.maximum(np.abs(caf), 1e-12))

    plt.figure(figsize=(12, 6))
    extent = [delays[0], delays[-1], dopplers[0], dopplers[-1]]
    plt.imshow(mag_db, aspect="auto", origin="lower", extent=extent)
    plt.colorbar(label="CAF Magnitude (dB)")
    plt.title("Cross Ambiguity Function |CAF(delay, doppler)|")
    plt.xlabel("Delay (samples)")
    plt.ylabel("Doppler (Hz)")

    # Plot injected truth FIRST (big red circle)
    if args.true_delay is not None and args.true_doppler is not None:
        plt.scatter(
            [args.true_delay], [args.true_doppler],
            marker="o",
            s=220,
            facecolors="none",
            edgecolors="red",
            linewidths=3.0,
            label="Injected truth",
            zorder=3
        )
        plt.text(
            args.true_delay, args.true_doppler,
            " truth",
            color="red",
            fontsize=10,
            zorder=5
        )

    # Plot estimated peak SECOND (big black X on top)
    plt.scatter(
        [pk_delay], [pk_fd],
        marker="x",
        s=200,
        c="black",
        linewidths=3.0,
        label="Estimated peak",
        zorder=4
    )
    plt.text(
        pk_delay, pk_fd,
        " peak",
        color="black",
        fontsize=10,
        zorder=6
    )

    plt.legend()
    plt.tight_layout()

    if args.save_png is not None:
        plt.savefig(args.save_png, dpi=300)
        print(f"Saved CAF heatmap to {args.save_png}")

    plt.show()



if __name__ == "__main__":
    main()
