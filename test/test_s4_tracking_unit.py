import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from soo_foundations.s4_tracking import TrackingConfig2D, _state_matrices, track_positions_cv2d  # noqa: E402


class TestS4TrackingUnit(unittest.TestCase):
    """Unit tests for Stage 4 constant-velocity tracking module."""

    def test_state_matrices_shape_and_layout(self):
        f, q, h = _state_matrices(dt_s=0.5, accel_sigma=2.0)
        self.assertEqual(f.shape, (4, 4))
        self.assertEqual(q.shape, (4, 4))
        self.assertEqual(h.shape, (2, 4))
        self.assertAlmostEqual(float(f[0, 2]), 0.5, places=9)
        self.assertAlmostEqual(float(f[1, 3]), 0.5, places=9)
        self.assertAlmostEqual(float(h[0, 0]), 1.0, places=9)
        self.assertAlmostEqual(float(h[1, 1]), 1.0, places=9)

    def test_empty_input_returns_empty_list(self):
        out = track_positions_cv2d(measured_xy=[])
        self.assertEqual(out, [])

    def test_tracking_reduces_rmse_vs_noisy_measurements(self):
        # Build deterministic synthetic constant-velocity motion + Gaussian measurement noise.
        rng = np.random.default_rng(123)
        dt = 1.0
        n = 80
        p0 = np.array([1000.0, -500.0], dtype=float)
        v = np.array([30.0, -20.0], dtype=float)
        true_xy = np.array([p0 + k * dt * v for k in range(n)], dtype=float)

        noise_sigma = 25.0
        meas_xy = true_xy + rng.normal(0.0, noise_sigma, size=true_xy.shape)

        cfg = TrackingConfig2D(
            dt_s=dt,
            process_accel_sigma_mps2=1.0,
            measurement_sigma_m=noise_sigma,
            init_pos_sigma_m=120.0,
            init_vel_sigma_mps=80.0,
        )
        out = track_positions_cv2d(measured_xy=meas_xy, cfg=cfg, x0_xy=true_xy[0], v0_xy=v)
        filt_xy = np.array([o.filtered_xy for o in out], dtype=float)

        rmse_meas = float(np.sqrt(np.mean(np.sum((meas_xy - true_xy) ** 2, axis=1))))
        rmse_filt = float(np.sqrt(np.mean(np.sum((filt_xy - true_xy) ** 2, axis=1))))

        self.assertEqual(len(out), n)
        self.assertLess(rmse_filt, rmse_meas)


if __name__ == "__main__":
    unittest.main()
