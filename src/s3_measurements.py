#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3 measurement summary utility.

Loads Stage 2 sweep CSV output, applies confidence/scoring filters, and prints
measurement statistics used before running positioning solvers.
"""

import argparse
import statistics

from soo_foundations.s3_positioning import MeasurementModelConfig, load_measurements_from_s2_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Stage 2 sweep CSV path")
    ap.add_argument("--samp_rate", type=float, default=1e6, help="sample rate in Hz")
    ap.add_argument("--carrier_hz", type=float, default=None, help="optional carrier for range-rate mapping")
    ap.add_argument("--min_confidence", type=float, default=0.0, help="minimum confidence filter")
    ap.add_argument("--scored_only", action="store_true", help="keep only rows marked scored=1")
    args = ap.parse_args()

    cfg = MeasurementModelConfig(carrier_hz=args.carrier_hz)
    meas = load_measurements_from_s2_csv(
        csv_path=args.csv,
        samp_rate=args.samp_rate,
        cfg=cfg,
        min_confidence=args.min_confidence,
        scored_only=args.scored_only,
    )

    if not meas:
        print("No measurements after filtering.")
        return

    conf = [m.confidence for m in meas]
    s_delay = [m.sigma_delay_samples for m in meas]
    s_fd = [m.sigma_doppler_hz for m in meas]

    print("Stage 3 Measurement Model Summary")
    print(f"  count                 = {len(meas)}")
    print(f"  confidence (mean)     = {statistics.mean(conf):.4f}")
    print(f"  confidence (median)   = {statistics.median(conf):.4f}")
    print(f"  sigma_delay_samp med  = {statistics.median(s_delay):.4f}")
    print(f"  sigma_doppler_hz med  = {statistics.median(s_fd):.4f}")
    if args.carrier_hz is not None:
        s_rr = [m.sigma_range_rate_mps for m in meas if m.sigma_range_rate_mps is not None]
        if s_rr:
            print(f"  sigma_rr_mps med      = {statistics.median(s_rr):.6f}")


if __name__ == "__main__":
    main()
