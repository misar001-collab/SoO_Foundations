import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


# Resolve repo paths once so every test can run from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def run_sweep(args):
    """
    Execute one `s2_validate_sweep` run and return `(stdout, csv_rows)`.

    The helper intentionally runs the module in a subprocess to validate real CLI
    behavior (argument parsing, summary output, and CSV emission), not just direct
    function calls.
    """
    with tempfile.TemporaryDirectory() as td:
        out_csv = Path(td) / "out.csv"

        # Build command with common deterministic flags.
        cmd = [
            sys.executable,
            "-m",
            "soo_foundations.s2_validate_sweep",
            *args,
            "--progress_every",
            "0",
            "--out_csv",
            str(out_csv),
        ]

        # Ensure subprocess imports package from local `src/` tree.
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        # `check=True` makes failures explicit and surfaces stderr in test failures.
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

        # Load all CSV rows as dicts for schema/field assertions.
        with out_csv.open("r", newline="") as f:
            rows = list(csv.DictReader(f))

    return proc.stdout, rows


class TestS2BehaviorUnit(unittest.TestCase):
    """Behavior-focused unit tests for tolerance and scoring policy choices."""

    def test_doppler_tol_max_uses_cpi_when_larger(self):
        # Very short duration forces CPI-derived tolerance to dominate bin-based tolerance.
        _, rows = run_sweep(
            [
                "--fd_cases",
                "1",
                "--duration",
                "0.001",
                "--true_delay",
                "100",
                "--delay_max",
                "900",
                "--doppler_tol_mode",
                "max",
                "--doppler_cpi_k",
                "0.5",
                "--pass_mode",
                "basic",
            ]
        )

        # Single case expected, with max(bin,cpi) selecting CPI contribution.
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["doppler_tol_source"], "max(bin,cpi)")
        self.assertAlmostEqual(float(row["doppler_tol_hz"]), 500.0, places=6)
        self.assertAlmostEqual(float(row["doppler_cpi_tol_hz"]), 500.0, places=6)

    def test_edge_guard_skips_edge_truth_cases(self):
        # Truth values at sweep boundaries should be marked near-edge and unscored.
        _, rows = run_sweep(
            [
                "--fd_start",
                "0",
                "--fd_stop",
                "80000",
                "--fd_cases",
                "3",
                "--duration",
                "0.02",
                "--fd_bins",
                "41",
                "--edge_guard_mode",
                "bin",
                "--edge_guard_bins",
                "1",
                "--pass_mode",
                "basic",
            ]
        )

        # Cases: [edge, center, edge] under this 3-point truth sweep.
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["truth_fd_near_edge"], "1")
        self.assertEqual(rows[0]["scored"], "0")
        self.assertEqual(rows[1]["truth_fd_near_edge"], "0")
        self.assertEqual(rows[1]["scored"], "1")
        self.assertEqual(rows[2]["truth_fd_near_edge"], "1")
        self.assertEqual(rows[2]["scored"], "0")

    def test_user_doppler_tol_override(self):
        # Explicit user tolerance should override auto tolerance modes.
        _, rows = run_sweep(
            [
                "--fd_cases",
                "1",
                "--duration",
                "0.02",
                "--doppler_tol_hz",
                "123.0",
                "--pass_mode",
                "basic",
            ]
        )

        row = rows[0]
        self.assertEqual(row["doppler_tol_source"], "user")
        self.assertAlmostEqual(float(row["doppler_tol_hz"]), 123.0, places=6)


if __name__ == "__main__":
    unittest.main()