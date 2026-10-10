#! /usr/bin/env python
# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""
Tests for tftb.processing.utils
"""

import unittest
import numpy as np
from tftb.processing import utils


class TestUtils(unittest.TestCase):

    def test_derive_window(self):
        """Test derivative of window function."""
        from scipy.signal.windows import gaussian
        g = gaussian(129, 10)
        dwindow = utils.derive_window(g)
        self.assertEqual(dwindow[64], 0)
        self.assertTrue(np.all(dwindow[:64] >= 0))
        self.assertTrue(np.all(dwindow[64:] <= 0))

    def test_derive_window_asymmetric_matches_matlab_dwindow(self):
        """The ramp term must match MATLAB dwindow: (h(hrow)-h(1))/(hrow-1).

        Symmetric windows have ramp=0 and can't catch a sign error in the ramp
        (this is why the bug went undetected). An asymmetric window exercises
        it: dwindow([0.1..0.9])' = [0.6, 0.1 x7, -0.4] (MATLAB-verified).
        """
        h = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        expected = np.array([0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -0.4])
        self.assertTrue(np.allclose(utils.derive_window(h), expected))


if __name__ == '__main__':
    unittest.main()
