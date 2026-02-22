import math
import sys
import unittest
from pathlib import Path

import numpy as np


# Add local package source path for direct helper imports in unit tests.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from soo_foundations.s2_validate_sweep import bin_width, in_grid, peak_to_median_db  # noqa: E402


class TestS2HelpersUnit(unittest.TestCase):
    """Small deterministic tests for helper utilities used by the sweep driver."""

    def test_bin_width_empty_or_single(self):
        # Degenerate grids have no spacing by definition.
        self.assertEqual(bin_width(np.array([])), 0.0)
        self.assertEqual(bin_width(np.array([42.0])), 0.0)

    def test_bin_width_regular_grid(self):
        # Endpoint spacing over (N-1) intervals should recover grid step.
        x = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)
        self.assertEqual(bin_width(x), 10.0)

    def test_in_grid_bounds(self):
        # Bounds are inclusive; values outside either side should fail.
        g = np.array([0.0, 5.0, 10.0], dtype=float)
        self.assertTrue(in_grid(0.0, g))
        self.assertTrue(in_grid(10.0, g))
        self.assertFalse(in_grid(-1.0, g))
        self.assertFalse(in_grid(11.0, g))

    def test_peak_to_median_db_finite(self):
        # Synthetic CAF with one strong cell should yield finite positive contrast.
        caf = np.array([[1 + 0j, 2 + 0j], [2 + 0j, 8 + 0j]], dtype=np.complex64)
        out = peak_to_median_db(caf)
        self.assertTrue(math.isfinite(out))
        self.assertGreater(out, 10.0)


if __name__ == "__main__":
    unittest.main()