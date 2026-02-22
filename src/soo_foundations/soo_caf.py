#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np


def caf_fftbank(
    s_ref: np.ndarray,
    s_surv: np.ndarray,
    samp_rate: float,
    delays: np.ndarray,
    dopplers: np.ndarray,
    nfft: int | None = None,
):
    """
    Compute Cross Ambiguity Function (CAF) over a delay/Doppler grid using an FFT bank.

    CAF definition used here (discrete-time form):

        CAF(fd, delay) = sum_n s_surv[n] * conj(s_ref[n-delay]) * exp(-j*2*pi*fd*n/fs)

    Direct computation is expensive for large grids. This implementation accelerates
    the delay dimension by using FFT-based correlation for each Doppler hypothesis.

    Returns:
        complex array of shape `(len(dopplers), len(delays))`
    """
    # Normalize inputs to expected complex dtype and matching length.
    s_ref = np.asarray(s_ref, dtype=np.complex64)
    s_surv = np.asarray(s_surv, dtype=np.complex64)

    n = min(len(s_ref), len(s_surv))
    s_ref = s_ref[:n]
    s_surv = s_surv[:n]

    # Delay and Doppler grids define output axes.
    delays = np.asarray(delays, dtype=int)
    dopplers = np.asarray(dopplers, dtype=float)

    # Correlation length is (2n - 1) for two length-n sequences.
    lin_len = 2 * n - 1

    # If not provided, choose next power-of-two FFT for speed.
    if nfft is None:
        nfft = 1 << int(np.ceil(np.log2(lin_len)))
    nfft = int(nfft)

    # Correlation via convolution trick:
    # corr(z, ref) == conv(z, conj(ref[::-1]))
    h = np.conj(s_ref[::-1])
    H = np.fft.fft(h, nfft)

    # Mapping from lag to linear-convolution index:
    # index i corresponds to lag = i - (n - 1)
    # so requested delay d maps to i = d + (n - 1)
    base = n - 1

    # Shared time vector used for Doppler phasor generation.
    t = np.arange(n, dtype=np.float64) / float(samp_rate)

    caf = np.empty((len(dopplers), len(delays)), dtype=np.complex64)

    # Doppler bank loop: one FFT-based correlation per Doppler hypothesis.
    for i, fd in enumerate(dopplers):
        # De-rotate surveillance by candidate Doppler.
        rot = np.exp(-1j * 2.0 * np.pi * fd * t).astype(np.complex64)
        z = s_surv * rot

        # FFT convolution for correlation.
        Z = np.fft.fft(z, nfft)
        y = np.fft.ifft(Z * H)
        y = y[:lin_len]  # discard circular tail and keep linear part

        # Extract only requested delay locations.
        idx = delays + base
        caf[i, :] = y[idx]

    return caf


def caf_peak(caf: np.ndarray, delays: np.ndarray, dopplers: np.ndarray):
    """
    Locate the global magnitude peak in a CAF matrix.

    Returns:
        `(peak_delay, peak_doppler_hz, complex_peak_value, peak_magnitude)`
    """
    mag = np.abs(caf)
    flat = np.argmax(mag)
    i, j = np.unravel_index(flat, mag.shape)
    return int(delays[j]), float(dopplers[i]), caf[i, j], float(mag[i, j])
