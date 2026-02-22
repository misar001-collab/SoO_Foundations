import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from s3_monte_carlo_eval import parse_noise_list  # noqa: E402
from soo_foundations.s3_evaluation import default_geometries_2d, evaluate_solver_monte_carlo  # noqa: E402


class TestS3EvaluationUnit(unittest.TestCase):
    """Unit tests for Stage 3 Step 3 evaluation helpers."""

    def test_parse_noise_list(self):
        out = parse_noise_list("1, 2,4.5")
        self.assertEqual(out, [1.0, 2.0, 4.5])

    def test_evaluation_output_schema(self):
        geoms = default_geometries_2d()[:1]
        rows = evaluate_solver_monte_carlo(
            geometries=geoms,
            noise_sigmas_m=[2.0, 6.0],
            runs_per_case=20,
            seed=5,
            max_iters=40,
            x0_xy=(0.0, 0.0),
        )
        self.assertEqual(len(rows), 2)
        required = {
            "geometry",
            "noise_sigma_m",
            "runs",
            "rmse_m",
            "median_err_m",
            "p95_err_m",
            "convergence_rate_pct",
        }
        for r in rows:
            self.assertTrue(required.issubset(set(r.keys())))
            self.assertGreater(float(r["rmse_m"]), 0.0)
            self.assertGreaterEqual(float(r["convergence_rate_pct"]), 0.0)
            self.assertLessEqual(float(r["convergence_rate_pct"]), 100.0)

    def test_rmse_increases_with_more_noise_on_average(self):
        # This is not mathematically guaranteed for every tiny sample size,
        # so we use enough runs to make trend violations unlikely.
        geoms = default_geometries_2d()[:1]
        rows = evaluate_solver_monte_carlo(
            geometries=geoms,
            noise_sigmas_m=[1.0, 10.0],
            runs_per_case=120,
            seed=9,
            max_iters=40,
            x0_xy=(0.0, 0.0),
        )
        rows = sorted(rows, key=lambda r: float(r["noise_sigma_m"]))
        rmse_lo = float(rows[0]["rmse_m"])
        rmse_hi = float(rows[1]["rmse_m"])
        self.assertLess(rmse_lo, rmse_hi)


if __name__ == "__main__":
    unittest.main()

