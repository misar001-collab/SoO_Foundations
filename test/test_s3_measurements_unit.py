import math
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from soo_foundations.s3_positioning import (  # noqa: E402
    MeasurementModelConfig,
    load_measurements_from_s2_csv,
    measurement_from_s2_row,
)


class TestS3MeasurementsUnit(unittest.TestCase):
    def test_measurement_conversion_and_range(self):
        row = {
            "delay_hat_samp": "1200",
            "fd_hat_hz": "50000",
            "doppler_bin_width_hz": "1000",
            "doppler_tol_hz": "500",
            "peak_to_median_db": "18",
            "scored": "1",
            "pass": "1",
            "peak_on_edge": "0",
            "peak_fd_near_edge": "0",
        }
        m = measurement_from_s2_row(row=row, samp_rate=1e6)
        self.assertAlmostEqual(m.delay_seconds, 1200.0 / 1e6, places=12)
        self.assertAlmostEqual(m.bistatic_range_m, m.delay_seconds * 299_792_458.0, places=6)
        self.assertTrue(m.sigma_delay_samples > 0.0)
        self.assertTrue(m.sigma_doppler_hz > 0.0)
        self.assertTrue(0.0 <= m.confidence <= 1.0)

    def test_confidence_penalty_for_unscored_and_edge(self):
        row_clean = {
            "delay_hat_samp": "1000",
            "fd_hat_hz": "10000",
            "doppler_bin_width_hz": "1000",
            "doppler_tol_hz": "500",
            "peak_to_median_db": "18",
            "scored": "1",
            "pass": "1",
            "peak_on_edge": "0",
            "peak_fd_near_edge": "0",
        }
        row_bad = dict(row_clean)
        row_bad["scored"] = "0"
        row_bad["peak_on_edge"] = "1"
        row_bad["pass"] = "0"

        m_clean = measurement_from_s2_row(row=row_clean, samp_rate=1e6)
        m_bad = measurement_from_s2_row(row=row_bad, samp_rate=1e6)
        self.assertGreater(m_clean.confidence, m_bad.confidence)

    def test_carrier_maps_doppler_to_range_rate(self):
        row = {
            "delay_hat_samp": "800",
            "fd_hat_hz": "1000",
            "doppler_bin_width_hz": "100",
            "doppler_tol_hz": "50",
            "peak_to_median_db": "15",
            "scored": "1",
            "pass": "1",
            "peak_on_edge": "0",
            "peak_fd_near_edge": "0",
        }
        cfg = MeasurementModelConfig(carrier_hz=1_000_000_000.0)
        m = measurement_from_s2_row(row=row, samp_rate=1e6, cfg=cfg)
        self.assertIsNotNone(m.bistatic_range_rate_mps)
        self.assertIsNotNone(m.sigma_range_rate_mps)
        expected = 1000.0 * (299_792_458.0 / 1_000_000_000.0)
        self.assertAlmostEqual(float(m.bistatic_range_rate_mps), expected, places=6)

    def test_load_measurements_filters(self):
        import csv
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s2.csv"
            with p.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "delay_hat_samp",
                        "fd_hat_hz",
                        "doppler_bin_width_hz",
                        "doppler_tol_hz",
                        "peak_to_median_db",
                        "scored",
                        "pass",
                        "peak_on_edge",
                        "peak_fd_near_edge",
                    ]
                )
                w.writerow(["1000", "10000", "1000", "500", "18", "1", "1", "0", "0"])
                w.writerow(["1000", "10000", "1000", "500", "8", "0", "0", "1", "1"])

            out = load_measurements_from_s2_csv(
                csv_path=p,
                samp_rate=1e6,
                min_confidence=0.2,
                scored_only=True,
            )
            self.assertEqual(len(out), 1)
            self.assertTrue(math.isfinite(out[0].bistatic_range_m))


if __name__ == "__main__":
    unittest.main()

