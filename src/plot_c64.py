#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_c64(path, max_samples=None):
    x = np.fromfile(path, dtype=np.complex64)
    if max_samples is not None and max_samples > 0:
        x = x[:max_samples]
    return x


def mag_db(x, eps=1e-12):
    return 20.0 * np.log10(np.maximum(np.abs(x), eps))


def fft_spectrum_db(x, samp_rate):
    """
    Returns (f_Hz, XdB) for fftshifted spectrum, in dB (magnitude).
    """
    n = len(x)
    if n == 0:
        return np.array([]), np.array([])

    w = np.hanning(n).astype(np.float64)
    xw = x.astype(np.complex128) * w

    X = np.fft.fftshift(np.fft.fft(xw))
    f = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / samp_rate))

    # Normalize for window gain (so levels are comparable-ish)
    X = X / np.sum(w)

    Xdb = mag_db(X)
    return f, Xdb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", default="data/original_gaussian.c64")
    p.add_argument("--reflected", default="data/reflected_gaussian.c64")
    p.add_argument("--samp_rate", type=float, default=1e6)
    p.add_argument("--samples", type=int, default=200000, help="0 = all (careful with big files)")
    p.add_argument("--fft_len", type=int, default=65536, help="FFT length (uses first fft_len samples)")
    p.add_argument("--show_spectrogram", action="store_true", help="Add spectrogram panels (best for chirps)")
    p.add_argument("--nfft", type=int, default=2048, help="Spectrogram NFFT")
    p.add_argument("--noverlap", type=int, default=1536, help="Spectrogram overlap")
    args = p.parse_args()

    max_samps = None if args.samples == 0 else args.samples
    orig = load_c64(args.original, max_samps)
    refl = load_c64(args.reflected, max_samps)

    n = min(len(orig), len(refl))
    orig = orig[:n]
    refl = refl[:n]

    if n == 0:
        raise SystemExit("No samples loaded. Check file paths and that files are non-empty.")

    t = np.arange(n) / args.samp_rate

    # --- FFT (use first fft_len samples to keep it fast)
    nfft = min(args.fft_len, n)
    f_o, Xo_db = fft_spectrum_db(orig[:nfft], args.samp_rate)
    f_r, Xr_db = fft_spectrum_db(refl[:nfft], args.samp_rate)

    # Layout
    if args.show_spectrogram:
        fig = plt.figure(figsize=(12, 14))
        gs_rows = 6
    else:
        fig = plt.figure(figsize=(12, 11))
        gs_rows = 4

    # 1) Original time (real)
    ax1 = plt.subplot(gs_rows, 1, 1)
    ax1.plot(t, orig.real)
    ax1.set_title("Original Signal (Real Component)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (normalized)")
    ax1.grid(True)

    # 2) Reflected time (real)
    ax2 = plt.subplot(gs_rows, 1, 2)
    ax2.plot(t, refl.real)
    ax2.set_title("Reflected Signal (Real Component)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude (normalized)")
    ax2.grid(True)

    # 3) Magnitude comparison
    ax3 = plt.subplot(gs_rows, 1, 3)
    ax3.plot(t, np.abs(orig), label="Original")
    ax3.plot(t, np.abs(refl), label="Reflected")
    ax3.set_title("Magnitude vs Time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Magnitude (normalized)")
    ax3.grid(True)
    ax3.legend()

    # 4) Frequency domain (FFT magnitude)
    ax4 = plt.subplot(gs_rows, 1, 4)
    ax4.plot(f_o, Xo_db, label=f"Original (N={nfft})")
    ax4.plot(f_r, Xr_db, label=f"Reflected (N={nfft})", alpha=0.85)
    ax4.set_title("Frequency Domain (FFT Magnitude, dB)")
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("Magnitude (dB)")
    ax4.grid(True)
    ax4.legend()

    if args.show_spectrogram:
        # 5) Spectrogram original
        ax5 = plt.subplot(gs_rows, 1, 5)
        Pxx, freqs, bins, im = ax5.specgram(
            orig, NFFT=args.nfft, Fs=args.samp_rate, noverlap=args.noverlap
        )
        ax5.set_title("Spectrogram (Original)")
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Frequency (Hz)")
        plt.colorbar(im, ax=ax5, label="Power (dB)")

        # 6) Spectrogram reflected
        ax6 = plt.subplot(gs_rows, 1, 6)
        Pxx2, freqs2, bins2, im2 = ax6.specgram(
            refl, NFFT=args.nfft, Fs=args.samp_rate, noverlap=args.noverlap
        )
        ax6.set_title("Spectrogram (Reflected)")
        ax6.set_xlabel("Time (s)")
        ax6.set_ylabel("Frequency (Hz)")
        plt.colorbar(im2, ax=ax6, label="Power (dB)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
