#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 1 chirp signal generator.

Creates a synthetic reference/surveillance pair using a chirp waveform, then
writes both streams to .c64 files for downstream CAF and positioning demos.
"""

import argparse
from pathlib import Path

from soo_foundations.soo_sim import generate_chirp_pair, write_c64


def main():
    # CLI entry for generating a chirp-based synthetic reference/surveillance pair.
    ap = argparse.ArgumentParser()

    # Sampling and duration control total number of waveform samples.
    ap.add_argument("--samp_rate", type=float, default=1e6, help="Hz")
    ap.add_argument("--duration", type=float, default=2.0, help="seconds")

    # Chirp configuration. In PowerShell, negative numeric values may need --f_start=-100e3.
    ap.add_argument("--f_start", type=float, default=-100e3, help="Hz")
    ap.add_argument("--f_end", type=float, default=100e3, help="Hz")
    ap.add_argument("--amp", type=float, default=1.0)

    # Reflection model controls: delay, Doppler, attenuation, and SNR-derived scaling.
    ap.add_argument("--delay", type=int, default=1200, help="samples")
    ap.add_argument("--doppler", type=float, default=50e3, help="Hz")
    ap.add_argument("--attn", type=float, default=0.9)
    ap.add_argument("--snr_db", type=float, default=10.0)

    # Destination folder for generated .c64 files.
    ap.add_argument("--out_dir", type=str, default="data")
    args = ap.parse_args()

    # Ensure output directory exists before writing binary waveform files.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate original chirp and reflected counterpart through shared simulation API.
    s_orig, s_ref = generate_chirp_pair(
        samp_rate=args.samp_rate,
        duration=args.duration,
        f_start=args.f_start,
        f_end=args.f_end,
        amp=args.amp,
        delay_samps=args.delay,
        doppler_hz=args.doppler,
        attn=args.attn,
        snr_db=args.snr_db,
    )

    # Persist outputs in interleaved complex64 format consumed by plotting/CAF scripts.
    p_orig = out_dir / "original_chirp.c64"
    p_ref = out_dir / "reflected_chirp.c64"
    write_c64(p_orig, s_orig)
    write_c64(p_ref, s_ref)

    # Emit concrete file paths and sample counts for quick verification.
    print(f"Wrote: {p_orig}  samples={len(s_orig)}")
    print(f"Wrote: {p_ref}   samples={len(s_ref)}")


if __name__ == "__main__":
    main()
