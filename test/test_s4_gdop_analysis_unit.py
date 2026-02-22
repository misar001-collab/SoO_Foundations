import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from s4_gdop_analysis import (  # noqa: E402
    geometry_metrics,
    parse_float_list,
    parse_xy,
    parse_xy_list,
    run_one_solve,
    unit_vec,
)


class TestS4GdopAnalysisUnit(unittest.TestCase):
    """Unit tests for Stage 4 GDOP analysis helpers."""

    def test_parse_xy_and_parse_xy_list(self):
        p = parse_xy("1.5, -2")
        self.assertTrue(np.allclose(p, np.array([1.5, -2.0], dtype=float)))

        pts = parse_xy_list("0,0; 10,1; -3,4")
        self.assertEqual(pts.shape, (3, 2))
        self.assertTrue(np.allclose(pts[1], np.array([10.0, 1.0], dtype=float)))

    def test_parse_float_list_and_unit_vec(self):
        vals = parse_float_list("1, 2.5, 3")
        self.assertEqual(vals, [1.0, 2.5, 3.0])

        uv = unit_vec(np.array([3.0, 4.0], dtype=float))
        self.assertTrue(np.allclose(uv, np.array([0.6, 0.8], dtype=float), atol=1e-12))

        with self.assertRaises(ValueError):
            unit_vec(np.array([0.0, 0.0], dtype=float))

    def test_geometry_gdop_worsens_near_baseline(self):
        # Same geometry, two targets: far from baseline vs close to baseline.
        txs = np.array([[0.0, 0.0], [16000.0, 2500.0], [-12000.0, 9500.0]], dtype=float)
        rx = np.array([3000.0, -2500.0], dtype=float)
        tx_base = txs[0]

        line = rx - tx_base
        u_line = line / np.linalg.norm(line)
        u_perp = np.array([-u_line[1], u_line[0]], dtype=float)
        anchor = tx_base + 0.45 * line

        target_far = anchor + 6000.0 * u_perp
        target_near = anchor + 200.0 * u_perp

        gdop_far, _ = geometry_metrics(target_far, txs, rx)
        gdop_near, _ = geometry_metrics(target_near, txs, rx)
        self.assertLess(gdop_far, gdop_near)

    def test_run_one_solve_returns_finite_estimate(self):
        rng = np.random.default_rng(7)
        txs = np.array([[0.0, 0.0], [16000.0, 2500.0], [-12000.0, 9500.0]], dtype=float)
        rx = np.array([3000.0, -2500.0], dtype=float)
        target = np.array([7000.0, 5200.0], dtype=float)

        ok, est = run_one_solve(
            rng=rng,
            target_true_xy=target,
            txs_xy=txs,
            rx_xy=rx,
            noise_sigma_m=5.0,
            max_iters=60,
            tol_step_m=1e-6,
        )
        self.assertTrue(ok)
        self.assertEqual(est.shape, (2,))
        self.assertTrue(np.all(np.isfinite(est)))
        self.assertLess(float(np.linalg.norm(est - target)), 80.0)

    def test_regression_near_baseline_rmse_is_worse(self):
        """
        Fixed-seed regression test for GDOP intuition.

        With same geometry/noise, near-baseline targets should produce noticeably
        larger Monte Carlo position RMSE than far-baseline targets.
        """
        txs = np.array([[0.0, 0.0], [16000.0, 2500.0], [-12000.0, 9500.0]], dtype=float)
        rx = np.array([3000.0, -2500.0], dtype=float)
        tx_base = txs[0]

        line = rx - tx_base
        u_line = line / np.linalg.norm(line)
        u_perp = np.array([-u_line[1], u_line[0]], dtype=float)
        anchor = tx_base + 0.45 * line

        target_far = anchor + 6000.0 * u_perp
        target_near = anchor + 200.0 * u_perp

        def rmse_for_target(target_xy, seed):
            rng = np.random.default_rng(seed)
            errs = []
            for _ in range(120):
                ok, est = run_one_solve(
                    rng=rng,
                    target_true_xy=target_xy,
                    txs_xy=txs,
                    rx_xy=rx,
                    noise_sigma_m=8.0,
                    max_iters=60,
                    tol_step_m=1e-6,
                )
                if ok and np.all(np.isfinite(est)):
                    errs.append(float(np.linalg.norm(est - target_xy)))
            e = np.asarray(errs, dtype=float)
            return float(np.sqrt(np.mean(e * e)))

        rmse_far = rmse_for_target(target_far, seed=123)
        rmse_near = rmse_for_target(target_near, seed=123)

        # Margin keeps the test strong while still robust to small numeric drift.
        self.assertGreater(rmse_near, 2.0 * rmse_far)


if __name__ == "__main__":
    unittest.main()
