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

"""

import unittest
from tftb.processing import time_domain as tmd
from tftb.generators import amplitude_modulated as am
from tftb.tests.test_base import TestBase


class TestTimeDomainProcessors(TestBase):

    def test_loctime(self):
        """Test computation of localized time characteristics."""
        signal = am.amgauss(160, 80, 50)
        tm, T = tmd.loctime(signal)
        self.assertAlmostEqual(tm, 79, places=6)
        self.assertAlmostEqual(T, 50, places=4)


if __name__ == '__main__':
    unittest.main()
