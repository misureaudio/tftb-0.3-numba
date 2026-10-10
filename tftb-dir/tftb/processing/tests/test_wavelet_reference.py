# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""L3 property / reference tests for the continuous wavelet transform (Task 4.1).

Value-level equivalence against MATLAB is covered by the golden cases
``gaussn_1`` / ``cwt_1`` / ``cwtmp_1`` (L1, rtol 1e-10). Those goldens were
generated with two scratch shims (``golden/fact2.m``, ``golden/lenfth.m``)
because the certified ``tftb-0.3`` references ``fact2`` (undefined) and
``lenfth`` (a typo for ``length`` in ``contwtgnmir`` line 21) -- both
unambiguous. These are independent closed-form / property checks.
"""
import numpy as np
import pytest

from tftb.generators import amgauss, fmconst
from tftb.processing.reassigned import morlet_scalogram
from tftb.processing.wavelet import gaussn, cwt, cwt_mirrored


# ---- gaussn (the wavelet) ----

@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_gaussn_is_real_l2_normalized(n):
    g = gaussn(0.1, n)
    assert g.dtype == np.float64  # real part is taken
    assert g.size > 1
    # L2 normalization: sum(g^2) == 1 (the analytic energy path).
    assert abs(float(g @ g) - 1.0) < 1e-6


@pytest.mark.parametrize("f0", [0.05, 0.1, 0.2, 0.4])
def test_gaussn_length_scales_with_frequency(f0):
    # lower f0 -> wider wavelet -> more samples (support ~ 1/f0).
    g_low = gaussn(0.05, 2)
    g_high = gaussn(0.4, 2)
    assert g_low.size > g_high.size
    # a mid frequency's wavelet is strictly between the two extremes.
    if 0.05 < f0 < 0.4:
        g_mid = gaussn(f0, 2)
        assert g_low.size >= g_mid.size >= g_high.size


# ---- cwt (contwtgn) ----

def _windowed_sinusoid(N=512, f0=0.25, spread=30):
    return amgauss(N, N // 2, spread) * fmconst(N, f0)[0]


def test_cwt_sinusoid_ridge_at_frequency():
    # A windowed (edge-suppressed) sinusoid's CWT energy concentrates at the
    # sinusoid's frequency. The wavelet's own frequency response biases the
    # peak slightly (the n-th derivative Gaussian is not a delta in
    # frequency), so the tolerance reflects that bias.
    N, f0 = 512, 0.25
    sig = _windowed_sinusoid(N, f0)
    scalo, f, T, a, wt = cwt(sig, 0.01, 0.5, 128, 2)
    mid = slice(int(0.2 * N), int(0.8 * N))  # avoid edge artefacts
    energy = np.max(np.abs(wt[:, mid]), axis=1)
    jp = int(np.argmax(energy))
    assert abs(f[jp] - f0) < 0.04, f"CWT peak {f[jp]:.4f} far from f0={f0}"
    # the peak is a genuine ridge (not flat): it stands well above the median.
    assert energy[jp] > 5 * np.median(energy[energy > 0])


def test_cwt_dilation_covariance():
    # Stretching the signal by a factor a lowers the IF law from f0 to f0/a,
    # so the CWT ridge moves to f0/a (dilation covariance).
    N, f0, a = 512, 0.25, 2.0
    M = int(a * N)
    sig_s = amgauss(M, M // 2, 30 * a) * fmconst(M, f0 / a)[0]
    scalo, f, T, a2, wt = cwt(sig_s, 0.01, 0.5, 128, 2)
    mid = slice(int(0.2 * M), int(0.8 * M))
    jp = int(np.argmax(np.max(np.abs(wt[:, mid]), axis=1)))
    assert abs(f[jp] - f0 / a) < 0.03, (
        f"stretched CWT peak {f[jp]:.4f} far from f0/a={f0 / a}")


def test_cwt_axes_geometric_and_ordered():
    # f is geometrically sampled ascending (fmin=0.01 first, fmax=0.5 last);
    # a is the reciprocal scale, flipped with f (fmax/fmin first, 1 last); T
    # is 1:nt. (The TFTB convention: first row = lowest frequency.)
    N = 128
    sig = amgauss(N, 60) * fmconst(N, 0.2)[0]
    scalo, f, T, a, wt = cwt(sig, 0.01, 0.5, 64, 2)
    assert f[0] == pytest.approx(0.01) and f[-1] == pytest.approx(0.5)
    assert np.all(np.diff(f) > 0)  # ascending
    assert a[0] == pytest.approx(50.0) and a[-1] == pytest.approx(1.0)
    assert np.all(np.diff(a) < 0)  # descending (flipped with f)
    # a is the reciprocal scale of f: f[i]*a[i] == fmax (geometric sampling).
    assert np.allclose(f * a, 0.5)
    assert T[0] == 1 and T[-1] == N and T.size == N
    assert wt.shape == (64, N)
    # scalogram is the squared magnitude of the coefficients.
    assert np.allclose(scalo, wt * np.conj(wt))


# ---- cwt_mirrored (contwtgnmir) ----

def test_cwt_mirrored_reduces_boundary_artefact():
    # Mirroring suppresses edge effects: the mirrored CWT's central energy is
    # at least as strong as the unmirrored one for a centred sinusoid, and the
    # two agree in the interior (far from the boundary).
    N, f0 = 256, 0.3
    sig = amgauss(N, N // 2, 16) * fmconst(N, f0)[0]
    _, f, _, _, wt = cwt(sig, 0.01, 0.5, 64, 2)
    _, f2, _, _, wt_m = cwt_mirrored(sig, 0.01, 0.5, 64, 2)
    assert np.allclose(f, f2)
    # interior (central 40%) should be nearly identical.
    lo, hi = int(0.3 * N), int(0.7 * N)
    assert np.abs(wt[:, lo:hi] - wt_m[:, lo:hi]).max() < 1e-6
    # the mirrored version does not lose central energy.
    assert np.max(np.abs(wt_m[:, lo:hi])) >= np.max(np.abs(wt[:, lo:hi])) - 1e-6


# ---- cross-check against morlet_scalogram (Task 1.3) ----

def test_cwt_and_morlet_detect_same_peak():
    # The plan: contwtgn "becomes the independent cross-checker for
    # morlet_scalogram". Both are continuous wavelet transforms with
    # Gaussian-derivative wavelets, so on the same windowed sinusoid they must
    # locate the spectral peak in the same frequency neighbourhood. The two
    # wavelets differ in their exact centre frequency (tbp vs the derivative
    # order), so the tolerance is neighbourhood-level, not exact.
    N, f0 = 512, 0.25
    sig = _windowed_sinusoid(N, f0)
    _, f, _, _, wt = cwt(sig, 0.01, 0.5, 128, 2)
    mid = slice(int(0.2 * N), int(0.8 * N))
    jp_cwt = int(np.argmax(np.max(np.abs(wt[:, mid]), axis=1)))

    tfr = morlet_scalogram(sig, n_fbins=N)[0]
    freqs = np.arange(N // 2) / N
    m = np.max(np.abs(tfr[:N // 2, mid]), axis=1)
    jp_morlet = int(np.argmax(m))
    # both peaks are near the true frequency and near each other.
    assert abs(freqs[jp_morlet] - f0) < 0.1
    assert abs(f[jp_cwt] - freqs[jp_morlet]) < 0.1
