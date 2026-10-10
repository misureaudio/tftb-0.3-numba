# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Reference / property tests for ``correlmx`` (Task 4.4a).

Value-level equivalence against MATLAB for all four algorithms is covered by
the golden case ``correlmx_1`` (rtol 1e-10). These are property checks:

* ``fbhermitian`` equals a naive direct (window-loop) computation.
* the ``fb*`` variants are exactly Hermitian (``Rx == conj(Rx.T)``).
* the AR(2) instantaneous-frequency estimate
  ``acos(-0.5*V(2,1)/V(1,1))/(2*pi)`` recovers the carrier of a noisy
  sinusoid (the ``correlmx.m`` docstring property, for both ``'burg'`` and
  ``'hermitian'``).
"""
import numpy as np
import pytest

from tftb.generators import fmconst
from tftb.processing.parametric import correlmx


def _naive_hermitian(x, p):
    """Direct window-loop accumulation of the Hermitian correlation matrix,
    mirroring ``correlmx.m`` (sum of outer products ``conj(v) v'``).  MATLAB
    ``vector`` is a column, so ``conj(vector)*vector.'`` is a 2-D outer
    product -> ``np.outer(conj(v), v)``."""
    L = x.shape[0]
    Rx = np.zeros((p + 1, p + 1), dtype=complex)
    for t in range(p + 1, L + 1):            # MATLAB t = p+2:L
        v = x[t - 1 - np.arange(p + 1)]      # MATLAB x(t-(0:p))
        Rx += np.outer(np.conj(v), v)
    return Rx / (L - p)


def _ar2_if(Rx):
    """AR(2) instantaneous frequency from the correlation matrix (the
    ``correlmx.m`` docstring formula):  acos(-0.5*V(2,1)/V(1,1))/(2*pi).

    The docstring uses the first ``eig`` column, which for these symmetric
    correlation matrices is the AR(2) *signal* mode -- the eigenvector of the
    **smallest**-magnitude eigenvalue (the AR residual direction carrying the
    carrier; the largest eigenvalue is the noise-floor mode and gives the
    wrong frequency).  We select by smallest |eigenvalue| so the result is
    independent of ``eig``'s column ordering.
    """
    w, V = np.linalg.eig(Rx)
    i = int(np.argmin(np.abs(w)))
    return float(np.arccos(-0.5 * V[1, i] / V[0, i]) / (2 * np.pi))


def test_correlmx_fbhermitian_matches_naive():
    rng = np.random.default_rng(0)
    N, p = 64, 3
    x = fmconst(N, 0.1)[0].real + 0.4 * rng.standard_normal(N)
    fast = correlmx(x, p, "fbhermitian")
    naive = _naive_hermitian(x, p)
    # fbhermitian = 0.5*(hermitian + conj(hermitian.T)); the naive matrix is
    # already Hermitian to round-off, so they must agree exactly.
    assert np.abs(fast - naive).max() <= 1e-12


def test_correlmx_fb_variants_are_hermitian():
    rng = np.random.default_rng(1)
    N, p = 48, 4
    x = fmconst(N, 0.15)[0].real + 0.3 * rng.standard_normal(N)
    for alg in ("fbhermitian", "fbburg"):
        Rx = correlmx(x, p, alg)
        assert np.abs(Rx - np.conj(Rx.T)).max() <= 1e-14, alg


def test_correlmx_burg_is_hermitian():
    # 'burg' is assembled as a Hermitian Toeplitz (first row = conj(first col)).
    rng = np.random.default_rng(2)
    N, p = 56, 5
    x = fmconst(N, 0.1)[0].real + 0.3 * rng.standard_normal(N)
    Rx = correlmx(x, p, "burg")
    assert np.abs(Rx - np.conj(Rx.T)).max() <= 1e-13


@pytest.mark.parametrize("alg", ["burg", "hermitian"])
def test_correlmx_ar2_if_recovers_carrier(alg):
    # The docstring property: a noisy 0.1-cosine's AR(2) model estimates the
    # carrier.  The eigenvector formula gives the IF in [0, 0.5]; a 0.1 carrier
    # is recovered to a few hundredths (the noise + AR(2) approximation).
    rng = np.random.default_rng(3)
    N = 200
    sig = fmconst(N, 0.1)[0].real + 0.4 * rng.standard_normal(N)
    f = _ar2_if(correlmx(sig, 2, alg))
    assert abs(f - 0.1) < 0.02, f"AR(2) IF {f:.4f} far from carrier 0.1 ({alg})"


def test_correlmx_unknown_algorithm_raises():
    x = fmconst(16, 0.1)[0]
    with pytest.raises(ValueError, match="unknown algorithm"):
        correlmx(x, 2, "notreal")


def test_correlmx_p_bounds():
    x = fmconst(16, 0.1)[0]
    with pytest.raises(ValueError, match="p must be greater than 0"):
        correlmx(x, 0, "burg")
    with pytest.raises(ValueError, match="greater than p"):
        correlmx(x, 17, "burg")   # p = L+1 > L
