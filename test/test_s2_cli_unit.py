import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


# Canonical project roots for subprocess-based CLI tests.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def run_cmd(args):
    """
    Run the sweep CLI with provided args and return CompletedProcess.

    Tests in this file intentionally verify argument-validation failures, so this
    helper does not use `check=True`.
    """
    with tempfile.TemporaryDirectory() as td:
        out_csv = Path(td) / "out.csv"

        # Common command skeleton with deterministic output location.
        cmd = [
            sys.executable,
            "-m",
            "soo_foundations.s2_validate_sweep",
            *args,
            "--out_csv",
            str(out_csv),
            "--progress_every",
            "0",
        ]

        # Inject local source tree into import path for subprocess execution.
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)


class TestS2CliUnit(unittest.TestCase):
    """CLI guardrail tests: invalid arguments should fail with clear messages."""

    def test_rejects_non_positive_duration(self):
        # Duration must be strictly positive.
        proc = run_cmd(["--duration", "0"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--duration must be > 0", proc.stderr)

    def test_rejects_non_positive_fd_cases(self):
        # At least one Doppler case is required for a sweep.
        proc = run_cmd(["--fd_cases", "0"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--fd_cases must be > 0", proc.stderr)

    def test_rejects_negative_edge_guard_bins(self):
        # Edge guard bin count cannot be negative.
        proc = run_cmd(["--edge_guard_bins", "-1"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--edge_guard_bins must be >= 0", proc.stderr)

    def test_rejects_negative_doppler_cpi_k(self):
        # CPI scaling factor for tolerance must be non-negative.
        proc = run_cmd(["--doppler_cpi_k", "-0.1"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--doppler_cpi_k must be >= 0", proc.stderr)


if __name__ == "__main__":
    unittest.main()