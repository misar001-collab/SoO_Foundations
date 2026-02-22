import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def run_cli(script_rel_path, args):
    """
    Run one Stage 4 CLI script and return CompletedProcess.

    Uses repo root as cwd and injects local `src` into PYTHONPATH so tests work
    without requiring editable install.
    """
    cmd = [sys.executable, str(REPO_ROOT / script_rel_path), *args]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env, check=True)


class TestS4CliUnit(unittest.TestCase):
    """CLI smoke tests for Stage 4 scripts."""

    def test_tracking_demo_writes_png(self):
        with tempfile.TemporaryDirectory() as td:
            out_png = Path(td) / "tracking.png"
            proc = run_cli(
                "src/s4_tracking_demo.py",
                [
                    "--no_show",
                    "--steps",
                    "20",
                    "--save_png",
                    str(out_png),
                ],
            )

            self.assertEqual(proc.returncode, 0)
            self.assertIn("Stage 4 Tracking Demo", proc.stdout)
            self.assertIn("rmse_raw_m", proc.stdout)
            self.assertIn("rmse_filtered_m", proc.stdout)
            self.assertTrue(out_png.exists())
            self.assertGreater(out_png.stat().st_size, 0)

    def test_gdop_analysis_writes_csv_and_png(self):
        with tempfile.TemporaryDirectory() as td:
            out_csv = Path(td) / "gdop.csv"
            out_png = Path(td) / "gdop.png"
            proc = run_cli(
                "src/s4_gdop_analysis.py",
                [
                    "--no_show",
                    "--runs_per_case",
                    "40",
                    "--baseline_distances_m",
                    "3000,1000,300",
                    "--out_csv",
                    str(out_csv),
                    "--save_png",
                    str(out_png),
                ],
            )

            self.assertEqual(proc.returncode, 0)
            self.assertIn("Wrote CSV:", proc.stdout)
            self.assertIn("Saved figure:", proc.stdout)
            self.assertTrue(out_csv.exists())
            self.assertTrue(out_png.exists())
            self.assertGreater(out_png.stat().st_size, 0)

            with out_csv.open("r", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)
            required_cols = {
                "distance_to_baseline_m",
                "gdop",
                "rmse_m",
                "blur_area_m2",
                "convergence_rate_pct",
            }
            self.assertTrue(required_cols.issubset(set(rows[0].keys())))


if __name__ == "__main__":
    unittest.main()
