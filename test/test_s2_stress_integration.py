import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


# Standard roots for subprocess-based integration checks.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def run_stress_case(duration_s: float):
    """
    Execute a fixed stress scenario and return `(scored_count, pass_count, pass_rate)`.

    The only swept variable in this helper is CPI duration; shorter durations reduce
    Doppler resolution, which should degrade robust pass behavior.
    """
    with tempfile.TemporaryDirectory() as td:
        out_csv = Path(td) / "stress.csv"

        cmd = [
            sys.executable,
            "-m",
            "soo_foundations.s2_validate_sweep",
            "--center_on_truth",
            "--snr_db",
            "20",
            "--duration",
            str(duration_s),
            "--true_delay",
            "1200",
            "--fd_start",
            "10000",
            "--fd_stop",
            "70000",
            "--fd_cases",
            "5",
            "--delay_half_window",
            "80",
            "--fd_half_window",
            "6000",
            "--fd_bins",
            "81",
            "--pass_mode",
            "robust",
            "--doppler_tol_mode",
            "max",
            "--doppler_cpi_k",
            "0.5",
            "--edge_guard_mode",
            "max",
            "--edge_guard_bins",
            "2",
            "--edge_guard_cpi_k",
            "1.0",
            "--progress_every",
            "0",
            "--out_csv",
            str(out_csv),
        ]

        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        # Require full command success; this is an integration correctness test.
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, capture_output=True, text=True, env=env)

        with out_csv.open("r", newline="") as f:
            rows = list(csv.DictReader(f))

    # Score only rows the validator marks as scorable.
    scored = [r for r in rows if r["scored"] == "1"]
    passed = [r for r in scored if r["pass"] == "1"]
    pass_rate = (len(passed) / len(scored)) if scored else 0.0
    return len(scored), len(passed), pass_rate


class TestS2StressIntegration(unittest.TestCase):
    """Stress-oriented integration tests around CPI-duration sensitivity."""

    def test_duration_shrink_degrades_pass_rate(self):
        # As duration shrinks, robust pass rate should not improve.
        _, _, rate_long = run_stress_case(0.2)
        _, _, rate_mid = run_stress_case(0.1)
        _, _, rate_short = run_stress_case(0.05)

        self.assertGreaterEqual(rate_long, rate_mid)
        self.assertGreaterEqual(rate_mid, rate_short)

    def test_long_duration_passes_and_short_duration_fails(self):
        # Under this configuration, long duration should fully pass; short should fully fail.
        scored_long, pass_long, rate_long = run_stress_case(0.2)
        scored_short, pass_short, rate_short = run_stress_case(0.05)

        self.assertEqual(scored_long, 5)
        self.assertEqual(pass_long, 5)
        self.assertEqual(rate_long, 1.0)

        self.assertEqual(scored_short, 5)
        self.assertEqual(pass_short, 0)
        self.assertEqual(rate_short, 0.0)


if __name__ == "__main__":
    unittest.main()