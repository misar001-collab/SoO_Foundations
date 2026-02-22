#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
soo_sim.py
Reusable synthetic signal generators for Stage 1 and Stage 2 validation.

Signal model used for the reflected/surveillance channel:

    s_ref[n] = attn * sigma(SNR) * s_orig[n-delay] * exp(j*2*pi*doppler*n/fs)

Where:
- `delay` shifts the original signal later in time (zero padded at the front).
- `doppler` applies a complex rotation per sample.
- `attn` scales reflected amplitude.
- `sigma(SNR)` maps requested SNR(dB) into the noise-scale convention used here.

All arrays are returned as `np.complex64` so they can be written directly to `.c64`
and consumed by the CAF pipeline without extra type conversion.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def snr_db_to_sigma(snr_db: float) -> float:
    """
    Convert SNR in dB to per-component Gaussian scale (`sigma`) for complex noise.

    Noise convention in this project:
        n = sigma * (randn + 1j * randn)

    Under this convention:
        E[|n|^2] = 2 * sigma^2

    If signal power is normalized near 1, then:
        SNR_linear = 1 / (2 * sigma^2)

    So:
        sigma = sqrt(1 / (2 * SNR_linear)).
    """
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    return math.sqrt(1.0 / (2.0 * snr_linear))


def apply_reflection_processing(
    s_orig: np.ndarray,
    samp_rate: float,
    delay_samps: int,
    doppler_hz: float,
    attn: float,
    snr_db: float,
) -> np.ndarray:
    """
    Apply the Stage-1 reflection model to produce a synthetic surveillance signal.

    Processing order is intentional and mirrors the analytical model:
    1) Delay by `delay_samps` using front zero-padding.
    2) Apply Doppler via complex exponential rotation.
    3) Apply attenuation and SNR-derived scaling.

    Output length exactly matches `s_orig` length.
    """
    # Keep computation in complex64 to match file format and avoid unnecessary casts.
    if s_orig.dtype != np.complex64:
        s_orig = s_orig.astype(np.complex64, copy=False)

    n = int(s_orig.shape[0])
    delay_samps = int(delay_samps)

    # Validate physically meaningful inputs early.
    if delay_samps < 0:
        raise ValueError("delay_samps must be >= 0")
    if samp_rate <= 0:
        raise ValueError("samp_rate must be > 0")

    # Delay stage: prepend zeros, then truncate back to original length.
    # This emulates a delayed receive path while preserving output size.
    if delay_samps == 0:
        s_del = s_orig
    else:
        z = np.zeros(delay_samps, dtype=np.complex64)
        s_del = np.concatenate([z, s_orig])[:n]

    # Doppler stage: rotate each sample by exp(j*2*pi*fd*t).
    # Using float64 time base keeps phase accumulation numerically stable.
    t = np.arange(n, dtype=np.float64) / float(samp_rate)
    rot = np.exp(1j * 2.0 * np.pi * float(doppler_hz) * t).astype(np.complex64)
    s_dopp = s_del * rot

    # Amplitude/SNR stage: project requested SNR onto the same convention
    # used in synthetic generation and downstream validation.
    sigma = snr_db_to_sigma(float(snr_db))
    s_ref = s_dopp * np.complex64(attn) * np.complex64(sigma)

    return s_ref.astype(np.complex64, copy=False)


def generate_gaussian_pair(
    samp_rate: float,
    n_samples: int,
    delay_samps: int,
    doppler_hz: float,
    attn: float,
    snr_db: float,
    seed: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate `(s_orig, s_ref)` where `s_orig` is complex Gaussian I/Q noise.

    `s_ref` is produced by passing `s_orig` through `apply_reflection_processing`.
    """
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    # Seeded RNG makes sweeps repeatable across runs when desired.
    rng = np.random.default_rng(seed)
    i = rng.standard_normal(n_samples)
    q = rng.standard_normal(n_samples)

    # Baseband complex Gaussian source.
    s_orig = (i + 1j * q).astype(np.complex64)

    s_ref = apply_reflection_processing(
        s_orig=s_orig,
        samp_rate=samp_rate,
        delay_samps=delay_samps,
        doppler_hz=doppler_hz,
        attn=attn,
        snr_db=snr_db,
    )
    return s_orig, s_ref


def generate_lfm_chirp(
    samp_rate: float,
    duration: float,
    f_start: float,
    f_end: float,
    amp: float = 1.0,
) -> np.ndarray:
    """
    Generate a complex baseband linear FM (LFM) chirp.

    Frequency law:
        f(t) = f_start + k*t,  k = (f_end - f_start) / T

    Phase law:
        phase(t) = 2*pi*(f_start*t + 0.5*k*t^2)

    Signal:
        x(t) = amp * exp(j*phase(t))
    """
    if samp_rate <= 0:
        raise ValueError("samp_rate must be > 0")
    if duration <= 0:
        raise ValueError("duration must be > 0")

    # Number of samples over requested duration at the configured sample rate.
    n = int(round(duration * samp_rate))

    # Build chirp phase from analytical integral of instantaneous frequency.
    t = np.arange(n, dtype=np.float64) / float(samp_rate)
    k = (float(f_end) - float(f_start)) / float(duration)
    phase = 2.0 * np.pi * (float(f_start) * t + 0.5 * k * t * t)
    x = float(amp) * np.exp(1j * phase)
    return x.astype(np.complex64)


def generate_chirp_pair(
    samp_rate: float,
    duration: float,
    f_start: float,
    f_end: float,
    delay_samps: int,
    doppler_hz: float,
    attn: float,
    snr_db: float,
    amp: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate `(s_orig, s_ref)` where `s_orig` is an LFM chirp waveform.

    This is the preferred synthetic pair for Stage-2 CAF validation because
    chirps create sharper ambiguity peaks than pure noise in many settings.
    """
    s_orig = generate_lfm_chirp(
        samp_rate=samp_rate,
        duration=duration,
        f_start=f_start,
        f_end=f_end,
        amp=amp,
    )
    s_ref = apply_reflection_processing(
        s_orig=s_orig,
        samp_rate=samp_rate,
        delay_samps=delay_samps,
        doppler_hz=doppler_hz,
        attn=attn,
        snr_db=snr_db,
    )
    return s_orig, s_ref


def write_c64(path: str | Path, x: np.ndarray) -> None:
    """
    Write a complex array to disk in raw interleaved `complex64` format (`.c64`).

    The directory tree is created automatically if it does not exist.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Enforce on-disk dtype expected by all readers in this repository.
    x.astype(np.complex64, copy=False).tofile(str(p))
