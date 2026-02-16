#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from pathlib import Path
import numpy as np


def snr_db_to_sigma(snr_db: float) -> float:
    # Same as your Gaussian stage:
    # snr_linear = 10**(snr_db/10); sigma = sqrt(1/(2*snr_linear))
    snr_linear = 10.0 ** (snr_db / 10.0)
    return math.sqrt(1.0 / (2.0 * snr_linear))


def generate_lfm_chirp(samp_rate: float, f_start: float, f_end: float, duration: float, amp: float = 1.0):
    """
    Complex baseband LFM chirp:
      f(t) = f_start + k*t,  k = (f_end - f_start)/T
      phase(t) = 2*pi*(f_start*t + 0.5*k*t^2)
      x(t) = amp * exp(j*phase(t))
    """
    n = int(round(duration * samp_rate))
    t = np.arange(n, dtype=np.float64) / samp_rate
    k = (f_end - f_start) / duration
    phase = 2.0 * np.pi * (f_start * t + 0.5 * k * t * t)
    x = amp * np.exp(1j * phase)
    return x.astype(np.complex64)


def apply_reflection_processing(
    x: np.ndarray,
    samp_rate: float,
    delay_samps: int,
    doppler_hz: float,
    attn: float,
    sigma: float,
) -> np.ndarray:
    """
    Mirrors your Gaussian reflected path:
      delay -> Doppler rotator -> attenuation -> multiply by sigma (SNR scaling)
    """
    n = x.shape[0]

    # Delay: prepend zeros, drop the tail to keep length constant
    delay_samps = int(delay_samps)
    if delay_samps < 0:
        raise ValueError("delay_samps must be >= 0")
    if delay_samps == 0:
        x_del = x
    else:
        z = np.zeros(delay_samps, dtype=np.complex64)
        x_del = np.concatenate([z, x])[:n]

    # Doppler: complex rotation exp(j*2*pi*fd*t)
    t = np.arange(n, dtype=np.float64) / samp_rate
    rot = np.exp(1j * 2.0 * np.pi * doppler_hz * t).astype(np.complex64)
    x_dopp = x_del * rot

    # Attenuation
    x_attn = x_dopp * np.complex64(attn)

    # Sigma scaling (SNR)
    x_out = x_attn * np.complex64(sigma)

    return x_out.astype(np.complex64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samp_rate", type=float, default=1e6, help="Hz")
    p.add_argument("--f_start", type=float, default=-100e3, help="Hz (use --f_start=-100e3 in PowerShell)")
    p.add_argument("--f_end", type=float, default=100e3, help="Hz")
    p.add_argument("--duration", type=float, default=2.0, help="seconds")
    p.add_argument("--amp", type=float, default=1.0, help="original chirp amplitude")

    # Reflected processing (mirrors Gaussian stage)
    p.add_argument("--delay", type=int, default=1200, help="delay in samples")
    p.add_argument("--doppler", type=float, default=50e3, help="Hz")
    p.add_argument("--attn", type=float, default=0.9)
    p.add_argument("--snr_db", type=float, default=10.0)

    p.add_argument("--out_dir", type=str, default="data", help="output directory")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_path = out_dir / "original_chirp.c64"
    reflected_path = out_dir / "reflected_chirp.c64"

    # Generate original
    x = generate_lfm_chirp(args.samp_rate, args.f_start, args.f_end, args.duration, args.amp)

    # Reflected processing
    sigma = snr_db_to_sigma(args.snr_db)
    y = apply_reflection_processing(
        x=x,
        samp_rate=args.samp_rate,
        delay_samps=args.delay,
        doppler_hz=args.doppler,
        attn=args.attn,
        sigma=sigma,
    )

    # Write files (complex64 interleaved)
    x.tofile(str(original_path))
    y.tofile(str(reflected_path))

    print(f"Wrote original:  {original_path}  samples={len(x)} bytes={original_path.stat().st_size}")
    print(f"Wrote reflected: {reflected_path} samples={len(y)} bytes={reflected_path.stat().st_size}")
    print(f"Reflected params: delay={args.delay} samps, doppler={args.doppler} Hz, attn={args.attn}, snr_db={args.snr_db}, sigma={sigma:.6g}")


if __name__ == "__main__":
    main()
