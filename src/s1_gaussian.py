#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 1 Gaussian signal generator.

Creates a synthetic reference/surveillance pair from Gaussian source data with
configurable delay, Doppler, attenuation, and noise settings.
"""

import argparse
from pathlib import Path

from soo_foundations.soo_sim import generate_gaussian_pair, write_c64


def main():
    # CLI entry for generating Gaussian-noise reference and reflected signal pair.
    ap = argparse.ArgumentParser()

    # Sampling config determines total number of generated samples.
    ap.add_argument("--samp_rate", type=float, default=1e6, help="Hz")
    ap.add_argument("--seconds", type=float, default=2.0, help="duration (s)")

    # Reflection model parameters used to synthesize the surveillance channel.
    ap.add_argument("--delay", type=int, default=1200, help="samples")
    ap.add_argument("--doppler", type=float, default=50e3, help="Hz")
    ap.add_argument("--attn", type=float, default=0.9)
    ap.add_argument("--snr_db", type=float, default=10.0)

    # Seed controls reproducibility of the Gaussian source.
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="data")
    args = ap.parse_args()

    # Convert time duration to integer sample count used by generator API.
    n_samples = int(round(args.seconds * args.samp_rate))

    # Ensure output location exists before writing raw complex files.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build original complex Gaussian stream and processed reflected stream.
    s_orig, s_ref = generate_gaussian_pair(
        samp_rate=args.samp_rate,
        n_samples=n_samples,
        delay_samps=args.delay,
        doppler_hz=args.doppler,
        attn=args.attn,
        snr_db=args.snr_db,
        seed=args.seed,
    )

    # Save signals in interleaved complex64 format for the rest of the pipeline.
    p_orig = out_dir / "original_gaussian.c64"
    p_ref = out_dir / "reflected_gaussian.c64"
    write_c64(p_orig, s_orig)
    write_c64(p_ref, s_ref)

    print(f"Wrote: {p_orig}  samples={len(s_orig)}")
    print(f"Wrote: {p_ref}   samples={len(s_ref)}")


if __name__ == "__main__":
    main()
