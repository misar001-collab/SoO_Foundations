#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3 - Step 1 measurement model for positioning inputs.

This module converts Stage 2 sweep outputs into structured measurements that
carry:
- delay and Doppler estimates
- derived physical quantities (delay seconds, bistatic range)
- uncertainty estimates
- confidence/weight terms for downstream solvers
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


SPEED_OF_LIGHT_MPS = 299_792_458.0


def _to_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    val = row.get(key, "")
    if val is None or str(val).strip() == "":
        return float(default)
    return float(val)


def _to_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    val = row.get(key, "")
    if val is None or str(val).strip() == "":
        return int(default)
    return int(float(val))


def _to_bool01(row: Dict[str, str], key: str, default: bool = False) -> bool:
    return bool(_to_int(row, key, 1 if default else 0))


@dataclass(frozen=True)
class MeasurementModelConfig:
    """
    Tunable parameters for converting Stage 2 rows into weighted measurements.

    The uncertainty model is intentionally simple and deterministic:
    - Doppler base uncertainty comes from grid/tolerance fields in Stage 2 CSV.
    - Delay base uncertainty starts at half-sample quantization.
    - Peak quality (peak_to_median_db) scales both uncertainties.
    """

    c_mps: float = SPEED_OF_LIGHT_MPS
    carrier_hz: Optional[float] = None

    # Peak-quality mapping parameters.
    p2m_db_ref: float = 8.0
    p2m_db_span: float = 20.0
    min_quality: float = 0.10
    max_quality: float = 1.00

    # Base uncertainty floors (before quality scaling).
    base_delay_sigma_samp: float = 0.50
    min_delay_sigma_samp: float = 0.25
    min_doppler_sigma_hz: float = 1.0

    # Confidence penalties for risky geometry/edge states.
    conf_penalty_unscored: float = 0.20
    conf_penalty_peak_edge: float = 0.60
    conf_penalty_fd_edge: float = 0.70
    conf_penalty_fail: float = 0.75


@dataclass(frozen=True)
class Stage3Measurement:
    """Structured Stage 3 measurement extracted from one Stage 2 row."""

    index: int
    delay_samples: float
    doppler_hz: float

    delay_seconds: float
    bistatic_range_m: float

    sigma_delay_samples: float
    sigma_delay_seconds: float
    sigma_range_m: float
    sigma_doppler_hz: float

    # If carrier_hz is provided, Doppler can be mapped to bistatic range-rate.
    bistatic_range_rate_mps: Optional[float]
    sigma_range_rate_mps: Optional[float]

    confidence: float
    weight_delay: float
    weight_doppler: float

    scored: bool
    passed: bool
    p2m_db: float

    row: Dict[str, str]


def quality_from_p2m_db(p2m_db: float, cfg: MeasurementModelConfig) -> float:
    """
    Map Stage 2 peak-to-median score (dB) into [min_quality, max_quality].

    `p2m_db_ref` is treated as the baseline for acceptable peaks.
    Values above it linearly increase quality up to `max_quality`.
    """
    raw = (float(p2m_db) - float(cfg.p2m_db_ref)) / float(cfg.p2m_db_span)
    q = float(cfg.min_quality) + max(0.0, min(1.0, raw)) * (float(cfg.max_quality) - float(cfg.min_quality))
    return max(float(cfg.min_quality), min(float(cfg.max_quality), q))


def measurement_from_s2_row(
    row: Dict[str, str],
    samp_rate: float,
    cfg: Optional[MeasurementModelConfig] = None,
    index: int = 0,
) -> Stage3Measurement:
    """Convert one Stage 2 CSV row into a weighted Stage 3 measurement."""
    if samp_rate <= 0.0:
        raise ValueError("samp_rate must be > 0")
    cfg = cfg or MeasurementModelConfig()

    delay_samples = _to_float(row, "delay_hat_samp")
    doppler_hz = _to_float(row, "fd_hat_hz")

    p2m_db = _to_float(row, "peak_to_median_db", cfg.p2m_db_ref)
    q = quality_from_p2m_db(p2m_db, cfg)

    # Delay uncertainty uses half-sample quantization scaled by quality.
    sigma_delay_samp = max(float(cfg.min_delay_sigma_samp), float(cfg.base_delay_sigma_samp) / math.sqrt(q))

    # Doppler uncertainty starts from bin/tolerance information when available.
    doppler_bin_w = abs(_to_float(row, "doppler_bin_width_hz", 0.0))
    doppler_tol = abs(_to_float(row, "doppler_tol_hz", 0.0))
    doppler_base = max(0.5 * doppler_bin_w, doppler_tol / 3.0, float(cfg.min_doppler_sigma_hz))
    sigma_doppler_hz = max(float(cfg.min_doppler_sigma_hz), doppler_base / math.sqrt(q))

    delay_seconds = delay_samples / float(samp_rate)
    bistatic_range_m = delay_seconds * float(cfg.c_mps)
    sigma_delay_seconds = sigma_delay_samp / float(samp_rate)
    sigma_range_m = sigma_delay_seconds * float(cfg.c_mps)

    # Optional conversion from Doppler to range-rate requires carrier frequency.
    if cfg.carrier_hz is not None and cfg.carrier_hz > 0.0:
        lamb = float(cfg.c_mps) / float(cfg.carrier_hz)
        bistatic_range_rate_mps = doppler_hz * lamb
        sigma_range_rate_mps = sigma_doppler_hz * lamb
    else:
        bistatic_range_rate_mps = None
        sigma_range_rate_mps = None

    scored = _to_bool01(row, "scored", default=True)
    passed = _to_bool01(row, "pass", default=True)
    peak_on_edge = _to_bool01(row, "peak_on_edge", default=False)
    peak_fd_near_edge = _to_bool01(row, "peak_fd_near_edge", default=False)

    confidence = q
    if not scored:
        confidence *= float(cfg.conf_penalty_unscored)
    if peak_on_edge:
        confidence *= float(cfg.conf_penalty_peak_edge)
    if peak_fd_near_edge:
        confidence *= float(cfg.conf_penalty_fd_edge)
    if not passed:
        confidence *= float(cfg.conf_penalty_fail)
    confidence = max(0.0, min(1.0, confidence))

    weight_delay = 1.0 / (sigma_delay_samp ** 2)
    weight_doppler = 1.0 / (sigma_doppler_hz ** 2)

    return Stage3Measurement(
        index=int(index),
        delay_samples=float(delay_samples),
        doppler_hz=float(doppler_hz),
        delay_seconds=float(delay_seconds),
        bistatic_range_m=float(bistatic_range_m),
        sigma_delay_samples=float(sigma_delay_samp),
        sigma_delay_seconds=float(sigma_delay_seconds),
        sigma_range_m=float(sigma_range_m),
        sigma_doppler_hz=float(sigma_doppler_hz),
        bistatic_range_rate_mps=bistatic_range_rate_mps,
        sigma_range_rate_mps=sigma_range_rate_mps,
        confidence=float(confidence),
        weight_delay=float(weight_delay),
        weight_doppler=float(weight_doppler),
        scored=bool(scored),
        passed=bool(passed),
        p2m_db=float(p2m_db),
        row=dict(row),
    )


def load_measurements_from_s2_csv(
    csv_path: str | Path,
    samp_rate: float,
    cfg: Optional[MeasurementModelConfig] = None,
    min_confidence: float = 0.0,
    scored_only: bool = False,
) -> List[Stage3Measurement]:
    """Load and convert all rows from a Stage 2 CSV file."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    out: List[Stage3Measurement] = []
    with path.open("r", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            meas = measurement_from_s2_row(row=row, samp_rate=samp_rate, cfg=cfg, index=i)
            if scored_only and not meas.scored:
                continue
            if meas.confidence < float(min_confidence):
                continue
            out.append(meas)
    return out

