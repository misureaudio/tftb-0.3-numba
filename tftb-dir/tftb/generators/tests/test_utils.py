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
Tests for tftb.generators.utils
"""

import unittest
import numpy as np

from tftb.generators import utils, fmlin


class TestUtils(unittest.TestCase):

    def test_sigmerge(self):
        """Test merging of signals with a given SNR."""
        signal = fmlin(128)[0]
        noise = np.random.randn(128,)
        gamma = 0.1
        x = utils.sigmerge(signal, noise, gamma)
        h_est = np.linalg.norm(signal) / np.linalg.norm(noise) * 10 ** (-gamma / 20)
        x_hat = signal + h_est * noise
        np.testing.assert_allclose(x, x_hat)


if __name__ == '__main__':
    unittest.main()
