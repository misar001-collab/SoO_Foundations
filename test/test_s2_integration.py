import csv
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


# Shared project roots for subprocess execution and local imports.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def run_sweep(args):
    """
    Run one end-to-end sweep execution and return `(stdout, parsed_rows)`.

    Integration tests use this helper to validate complete CLI behavior and
    emitted CSV content together.
    """
    with tempfile.TemporaryDirectory() as td:
        out_csv = Path(td) / "stage2.csv"

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

        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, capture_output=True, text=True, env=env)

        with out_csv.open("r", newline="") as f:
            rows = list(csv.DictReader(f))

    return proc.stdout, rows


def summary_int(stdout: str, key: str) -> int:
    """
    Extract integer value from summary block lines like: `key = 123`.

    Raises AssertionError with full stdout when a key is missing so failures are
    immediately diagnosable.
    """
    m = re.search(rf"^\s*{re.escape(key)}\s*=\s*(\d+)", stdout, flags=re.MULTILINE)
    if not m:
        raise AssertionError(f"summary key '{key}' not found in output:\n{stdout}")
    return int(m.group(1))


class TestS2Integration(unittest.TestCase):
    """End-to-end integration tests for Stage-2 sweep logic and reporting."""

    def test_fast_end_to_end_csv_schema(self):
        # Quick smoke run verifies command path and essential CSV columns.
        stdout, rows = run_sweep(["--duration", "0.01", "--fd_cases", "2", "--fd_bins", "21"])

        self.assertIn("Stage 2 Sweep Summary", stdout)
        self.assertIn("scored", stdout)
        self.assertEqual(len(rows), 2)

        row = rows[0]
        expected_cols = [
            "fd_true_hz",
            "fd_hat_hz",
            "fd_err_hz",
            "delay_true_samp",
            "delay_hat_samp",
            "delay_err_samp",
            "doppler_bin_width_hz",
            "doppler_cpi_tol_hz",
            "doppler_tol_source",
            "doppler_tol_hz",
            "doppler_edge_guard_hz",
            "doppler_edge_guard_source",
            "truth_fd_near_edge",
            "peak_fd_near_edge",
            "scored",
            "pass",
        ]
        for col in expected_cols:
            self.assertIn(col, row)

    def test_must_pass_delay_sweep_safe_config(self):
        # High-SNR, truth-centered local windows should robustly pass across delays.
        test_delays = [200, 600, 1000, 1400]
        for d in test_delays:
            _, rows = run_sweep(
                [
                    "--center_on_truth",
                    "--snr_db",
                    "60",
                    "--duration",
                    "0.2",
                    "--fd_start",
                    "0",
                    "--fd_stop",
                    "0",
                    "--fd_cases",
                    "1",
                    "--true_delay",
                    str(d),
                    "--delay_half_window",
                    "80",
                    "--fd_half_window",
                    "6000",
                    "--fd_bins",
                    "121",
                    "--pass_mode",
                    "robust",
                ]
            )

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["scored"], "1")
            self.assertEqual(row["pass"], "1")
            self.assertEqual(row["fail_reason"], "none")

    def test_must_pass_doppler_sweep_safe_config(self):
        # Same safe setup across a Doppler range should produce all passes.
        stdout, rows = run_sweep(
            [
                "--center_on_truth",
                "--snr_db",
                "60",
                "--duration",
                "0.2",
                "--true_delay",
                "1000",
                "--fd_start",
                "5000",
                "--fd_stop",
                "75000",
                "--fd_cases",
                "9",
                "--delay_half_window",
                "80",
                "--fd_half_window",
                "6000",
                "--fd_bins",
                "121",
                "--pass_mode",
                "robust",
            ]
        )

        self.assertEqual(len(rows), 9)
        self.assertEqual(summary_int(stdout, "cases"), 9)
        self.assertEqual(summary_int(stdout, "scored"), 9)
        self.assertEqual(summary_int(stdout, "pass"), 9)
        self.assertEqual(summary_int(stdout, "fail"), 0)
        self.assertTrue(all(r["pass"] == "1" for r in rows))

    def test_edge_guard_accounting_matches_csv(self):
        # Summary counters should match row-level scored/edge flags in CSV.
        stdout, rows = run_sweep(
            [
                "--fd_start",
                "0",
                "--fd_stop",
                "80000",
                "--fd_cases",
                "5",
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

        cases = summary_int(stdout, "cases")
        scored = summary_int(stdout, "scored")
        skipped = summary_int(stdout, "skipped")
        self.assertEqual(cases, len(rows))
        self.assertEqual(scored + skipped, cases)

        scored_csv = sum(1 for r in rows if r["scored"] == "1")
        self.assertEqual(scored_csv, scored)

    def test_pass_mode_changes_final_pass_column(self):
        # Construct a case where `pass_basic` can be true but robust mode rejects edge peaks.
        base_args = [
            "--fd_start",
            "0",
            "--fd_stop",
            "0",
            "--fd_cases",
            "1",
            "--duration",
            "0.02",
            "--fd_bins",
            "41",
            "--edge_guard_mode",
            "bin",
            "--edge_guard_bins",
            "0",
            "--snr_db",
            "60",
            "--delay_tol",
            "5000",
            "--doppler_tol_hz",
            "100000",
        ]

        _, rows_basic = run_sweep([*base_args, "--pass_mode", "basic"])
        _, rows_robust = run_sweep([*base_args, "--pass_mode", "robust"])

        self.assertEqual(rows_basic[0]["pass_basic"], "1")
        self.assertEqual(rows_robust[0]["pass_basic"], "1")
        self.assertEqual(rows_basic[0]["pass"], "1")
        self.assertEqual(rows_robust[0]["pass"], "0")
        self.assertEqual(rows_robust[0]["peak_on_edge"], "1")

    def test_user_doppler_tol_override_integration(self):
        # User-provided tolerance should propagate to both summary text and CSV fields.
        stdout, rows = run_sweep(["--fd_cases", "1", "--doppler_tol_hz", "123.0", "--pass_mode", "basic"])
        self.assertIn("(user specified)", stdout)
        self.assertEqual(rows[0]["doppler_tol_source"], "user")
        self.assertAlmostEqual(float(rows[0]["doppler_tol_hz"]), 123.0, places=6)


if __name__ == "__main__":
    unittest.main()