import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from soo_foundations.s3_solver import (  # noqa: E402
    make_bistatic_measurements_2d,
    predict_bistatic_range_2d,
    solve_position_bistatic_wls_2d,
)


class TestS3SolverUnit(unittest.TestCase):
    """Unit tests for Stage 3 2D bistatic least-squares solver."""

    def test_predict_bistatic_range_basic_geometry(self):
        # Simple right-angle geometry where distances are easy to verify by hand.
        tx = [0.0, 0.0]
        rx = [10.0, 0.0]
        target = [5.0, 0.0]
        # |target-tx|=5 and |target-rx|=5, so total should be 10.
        r = predict_bistatic_range_2d(target, tx, rx)
        self.assertAlmostEqual(r, 10.0, places=9)

    def test_solver_recovers_noiseless_target(self):
        # In noiseless conditions, least-squares should recover true point closely.
        tx = np.array([0.0, 0.0], dtype=float)
        rxs = np.array([[12000.0, 1000.0], [-7000.0, 11000.0], [9000.0, -9000.0], [16000.0, 15000.0]], dtype=float)
        true_xy = np.array([4000.0, 6000.0], dtype=float)

        ranges = [predict_bistatic_range_2d(true_xy, tx, rx) for rx in rxs]
        meas = make_bistatic_measurements_2d(
            measured_ranges_m=ranges,
            tx_positions_xy=[tx for _ in rxs],
            rx_positions_xy=rxs,
            sigma_ranges_m=[1.0] * len(ranges),
            confidences=[1.0] * len(ranges),
        )

        out = solve_position_bistatic_wls_2d(measurements=meas, x0_xy=[0.0, 0.0], max_iters=60)
        est = np.array(out.estimated_xy, dtype=float)
        err = float(np.linalg.norm(est - true_xy))

        self.assertTrue(out.success)
        self.assertLess(err, 1e-3)
        self.assertLess(out.residual_rms_m, 1e-3)

    def test_solver_handles_moderate_noise(self):
        # With moderate Gaussian noise, solver should still land reasonably close.
        rng = np.random.default_rng(1234)
        tx = np.array([0.0, 0.0], dtype=float)
        rxs = np.array([[14000.0, 0.0], [0.0, 15000.0], [-12000.0, 6000.0], [7000.0, -13000.0]], dtype=float)
        true_xy = np.array([6500.0, 3500.0], dtype=float)

        sigma = 6.0
        ranges = []
        for rx in rxs:
            r_true = predict_bistatic_range_2d(true_xy, tx, rx)
            ranges.append(float(r_true + rng.normal(0.0, sigma)))

        meas = make_bistatic_measurements_2d(
            measured_ranges_m=ranges,
            tx_positions_xy=[tx for _ in rxs],
            rx_positions_xy=rxs,
            sigma_ranges_m=[sigma] * len(ranges),
            confidences=[1.0] * len(ranges),
        )

        out = solve_position_bistatic_wls_2d(measurements=meas, x0_xy=[0.0, 0.0], max_iters=60)
        est = np.array(out.estimated_xy, dtype=float)
        err = float(np.linalg.norm(est - true_xy))

        self.assertTrue(out.success)
        # Threshold chosen to be realistic for this synthetic geometry/noise setup.
        self.assertLess(err, 80.0)


if __name__ == "__main__":
    unittest.main()

