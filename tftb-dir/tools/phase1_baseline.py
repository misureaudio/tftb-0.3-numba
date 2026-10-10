# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Phase 1 baseline: directly exercise the 9 known-broken functions.

The inherited 0.2.0 suite (57 tests) does not cover these; they were flagged
in the earlier manual smoke test. This establishes the real Phase 1 work list
in the workspace venv (numpy 2.4.6, scipy 1.17.1).
"""
import numpy as np
import traceback

N = 128
x = np.real(np.exp(-(np.arange(N) - 60) ** 2 / (2 * 30 ** 2)) *
            np.exp(2j * np.pi * np.arange(N) * (0.1 + 0.02 * np.arange(N))))

checks = []

def record(name, fn):
    try:
        out = fn()
        checks.append((name, "OK", getattr(out, "shape", type(out).__name__)))
    except Exception as e:
        checks.append((name, "FAIL", f"{type(e).__name__}: {e}"))
        tb = traceback.format_exc().splitlines()
        checks.append((name + "  >>last", "", tb[-1].strip()))

import tftb.processing as P
from tftb.processing import (
    WignerVilleDistribution, PseudoWignerVilleDistribution, Spectrogram,
    MargenauHillDistribution,
)
from tftb.processing import reassigned_spectrogram, reassigned_smoothed_pseudo_wigner_ville
from tftb.processing.affine import Scalogram, BertrandDistribution
from tftb.processing.linear import ShortTimeFourierTransform
from tftb.processing.cohen import smoothed_pseudo_wigner_ville

# 1-6: reassigned module (functions, not classes)
record("reassigned.spectrogram", lambda: reassigned_spectrogram(x))
record("reassigned.spwv", lambda: reassigned_smoothed_pseudo_wigner_ville(x))

# 7: gabor (linear)
from tftb.processing.linear import gabor
record("linear.gabor", lambda: gabor(x, 32, 2))

# 8: scale (generators utils) — scale(X, a, fmin, fmax, N); hardcoded to T=128,N=128
from tftb.generators import scale
record("generators.scale", lambda: scale(np.arange(128, dtype=float), 0.5, 0.1, 0.4, 128))

# 9: affine smoothed_pseudo_wigner (if present)
try:
    from tftb.processing.affine import smoothed_pseudo_wigner
    record("affine.spw", lambda: smoothed_pseudo_wigner(x))
except ImportError:
    checks.append(("affine.spw", "ABSENT", "not in 0.2.0 affine"))

print(f"{'function':28} {'status':12} detail")
print("-" * 70)
for name, status, detail in checks:
    print(f"{name:28} {status:12} {detail}")
