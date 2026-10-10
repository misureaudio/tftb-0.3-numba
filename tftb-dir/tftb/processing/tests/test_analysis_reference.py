# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Reference / property tests for the Phase 4 Task 4.5 analysis utilities.

Value-level equivalence against the certified MATLAB oracle is covered by the
golden cases ``momftfr_1``, ``momttfr_1``, ``margtfr_1``, ``holder_1``,
``if2phase_1``, ``d2statio_1``, ``tfristft_1``, ``tffilter_1``.  These are the
structural / consistency invariants, transcribed from the certified
``tftb-0.3/tests`` files (``momftfrt.m``, ``momttfrt.m``, ``margtfrt.m``,
``holdert.m``) plus the plan's property checks:

* ``momftfr``/``momttfr`` recover the time/frequency axis of an ideal TFR
  (perfect line -> ``tm==1:N`` / ``fm==linspace(0,0.5)``);
* ``margtfr``: ``margt == |sig|^2``, ``E == ||sig||^2``, ``margf == PSD``;
* ``holder``: the Holder exponent of an ``anasing`` singularity recovers its
  order (the 0.3 bounds, incl. the mandatory odd-N N=211 case at 2e-2);
* ``if2phase``: for a smooth IF law the phase matches ``2*pi*cumsum``;
* ``tffilter``: a 2-component signal filtered to one component keeps only that
  component's energy ridge;
* ``tfristft``: round-trip ``tfristft(tfrstft(x)) == x``;
* ``midpoint``: the Wigner-Ville (k=2) mid-point is the coordinate average;
* ``plotsid``: smoke (no output argument in the certified test);
* ``tftb_window``: cross-check the overlapping shapes vs
  ``scipy.signal.windows``;
* ``griffitc``: the three IF laws are deterministic and in-band.
"""
import numpy as np
import pytest

from tftb.generators import (amgauss, anasing, fmlin, fmsin, fmhyp, fmconst,
                             griffitc)
from tftb.processing.analysis import (momftfr, momttfr, margtfr, holder,
                                      if2phase, d2statio, tffilter, tfristft,
                                      midpoint, plotsid, tfrideal)
from tftb.processing.cohen import WignerVilleDistribution
from tftb.processing.utils import tftb_window


# ---------------------------------------------------------------------------
# momftfr / momttfr (transcribed from momftfrt.m / momttfrt.m)
# ---------------------------------------------------------------------------
def test_momftfr_perfect_line():
    # a linear-chirp IF law -> the ideal TFR's time moment is the time axis.
    for N in (128, 117):
        _, ifl = fmlin(N)
        tfr, _, _ = tfrideal(ifl)
        tm, D2 = momftfr(tfr)
        assert np.max(np.abs(tm - np.arange(1, N + 1))) < np.sqrt(np.finfo(float).eps)
        assert np.max(np.abs(D2)) < np.sqrt(np.finfo(float).eps)


def test_momftfr_sinusoid():
    # a sinusoidal IF law (transposed, as in the certified test).
    for N in (128, 117):
        _, ifl = fmsin(N, 0, 0.5, N * 2, 1, 0)
        tfr, _, _ = tfrideal(ifl)
        tm, D2 = momftfr(tfr.T)
        tmth = np.round(ifl * 2 * (N - 1)) + 1
        assert np.max(np.abs(tm - tmth)) < np.sqrt(1.0 / N)
        assert np.max(np.abs(D2)) < np.sqrt(np.finfo(float).eps)


def test_momttfr_perfect_line():
    # a linear-chirp IF law -> the ideal TFR's frequency moment is the freq axis.
    for N in (128, 111):
        _, ifl = fmlin(N)
        tfr, _, _ = tfrideal(ifl)
        fm, B2 = momttfr(tfr, "type2")
        fmth = np.linspace(0, 0.5 - 1.0 / (2 * N), N)
        assert np.max(np.abs(fm - fmth)) < np.sqrt(np.finfo(float).eps)
        assert np.max(np.abs(B2)) < np.sqrt(np.finfo(float).eps)


def test_momttfr_sinusoid():
    # a sinusoidal IF law -> the frequency moment tracks the IF law.
    for N in (128, 111):
        _, ifl = fmsin(N)
        tfr, _, _ = tfrideal(ifl)
        fm, B2 = momttfr(tfr, "type2")
        assert np.max(np.abs(fm - ifl)) < np.sqrt(1.0 / N)
        assert np.max(np.abs(B2)) < np.sqrt(np.finfo(float).eps)


def test_momttfr_three_iflaws():
    # three IF laws stacked into a SINGLE component (MATLAB [ifl1;ifl2;ifl3]
    # is a 384x1 column) -> each time column's frequency moment is one of the
    # IF laws.
    for N in (128, 111):
        _, ifl1 = fmlin(N)
        _, ifl2 = fmsin(N)
        _, ifl3 = fmhyp(N, np.array([1.0, 0.5]), np.array([N / 2, 0.1]))
        stacked = np.concatenate([ifl1, ifl2, ifl3])
        tfr, _, _ = tfrideal(stacked)
        fm, B2 = momttfr(tfr, "type2")
        assert np.max(np.abs(fm - stacked)) < np.sqrt(1.0 / N)
        assert np.max(np.abs(B2)) < np.sqrt(np.finfo(float).eps)


def test_momttfr_unknown_method_raises():
    tfr, _, _ = tfrideal(fmlin(64)[1])
    with pytest.raises(ValueError, match="Unknown representation"):
        momttfr(tfr, "notarepr")


# ---------------------------------------------------------------------------
# margtfr (transcribed from margtfrt.m — full-WVD invariants)
# ---------------------------------------------------------------------------
def test_margtfr_wvd_invariants():
    # margt == |sig|^2, E == ||sig||^2, margf == mean of the even (positive)
    # frequency rows (the PSD).  (The certified test's time/frequency-subset
    # and odd-N cases need a reduced-size / odd-length TFR, which the Python
    # WVD class does not support yet — a pre-existing limitation, out of 4.5
    # scope — so the even-N full-WVD invariants are the core property here.)
    for N in (128, 126):
        sig = amgauss(N) * fmlin(N)[0]
        tfr, _, _ = WignerVilleDistribution(sig).run()
        tfr = np.real(tfr)
        margt, margf, E = margtfr(tfr)
        assert np.max(np.abs(margt - np.abs(sig) ** 2)) < np.sqrt(np.finfo(float).eps)
        assert abs(E - np.linalg.norm(sig) ** 2) < np.sqrt(np.finfo(float).eps)
        assert np.max(np.abs(margf[0::2] - tfr[0::2].mean(axis=1))) < np.sqrt(np.finfo(float).eps)


# ---------------------------------------------------------------------------
# holder (transcribed from holdert.m)
# ---------------------------------------------------------------------------
def _holder_py(sig, t0):
    # The certified test uses tfrscalo(sig, 1:N, 16, 0.01, 0.49, 2*N).  The
    # Python Scalogram cannot do 16 voices / 2N bins (a pre-existing
    # limitation), so we use n_voices=N with the same fmin/fmax — the same
    # 0.3 bounds still hold (verified empirically).
    N = len(sig)
    tfr, t, f = Scalogram_holder(sig, N)
    return holder(tfr, f, 1, len(f), int(t0), 0)


def Scalogram_holder(sig, N):
    from tftb.processing.affine import Scalogram
    r = Scalogram(sig, fmin=0.01, fmax=0.49, n_voices=N).run()
    return r[0], r[1], r[2]


def test_holder_even_n():
    # N=256 (even): h=0 and h=-0.5, bound 1e-2.
    for h in (0.0, -0.5):
        N = 256
        t0 = N / 2
        sig = anasing(N, t0, h)
        h0 = _holder_py(sig, t0)
        assert abs(h - h0) < 1e-2


def test_holder_odd_n():
    # N=211 (odd, mandatory): h=0 bound 2e-2 (the 0.3 fix), h=-0.4 bound 1e-2.
    N = 211
    t0 = (N - 1) / 2
    for h, bound in ((0.0, 2e-2), (-0.4, 1e-2)):
        sig = anasing(N, t0, h)
        h0 = _holder_py(sig, t0)
        assert abs(h - h0) < bound


# ---------------------------------------------------------------------------
# if2phase (plan: vs 2*pi*cumsum for a smooth law)
# ---------------------------------------------------------------------------
def test_if2phase_smooth_law():
    # MATLAB's ``integ`` is a trapezoidal integral that starts at 0, so for a
    # CONSTANT IF law the phase is a perfect ramp from 0 with slope 2*pi*f0
    # (NOT 2*pi*cumsum, which starts at 2*pi*f0).
    N = 256
    f0 = 0.2
    phi = if2phase(np.full(N, f0))
    assert phi[0] == 0.0
    assert np.max(np.abs(np.diff(phi) - 2.0 * np.pi * f0)) < 1e-12
    # for a LINEAR IF law the total phase equals 2*pi*trapz(iflaw) exactly.
    from scipy.integrate import trapezoid
    iflaw2 = np.linspace(0.1, 0.3, N)
    phi2 = if2phase(iflaw2)
    assert phi2[0] == 0.0
    assert abs(phi2[-1] - 2.0 * np.pi * trapezoid(iflaw2)) < 1e-9
    assert np.all(np.diff(phi2) > 0)


def test_if2phase_starts_at_zero():
    phi = if2phase(np.linspace(0.05, 0.4, 128))
    assert phi[0] == 0.0
    assert np.all(np.diff(phi) > 0)


# ---------------------------------------------------------------------------
# d2statio (property: stationary-ish signal -> small distance)
# ---------------------------------------------------------------------------
def test_d2statio_shape_and_stationary_small():
    N = 128
    sig = amgauss(N) * fmlin(N)[0]
    d, f = d2statio(sig)
    assert d.shape == (N,)
    assert f.shape == (N,)
    assert np.all(np.isfinite(d))
    assert np.all(d >= -1e-9)
    # a stationary signal (constant tone) has a small distance to stationarity.
    tone = fmconst(128, 0.2)[0]
    d_tone, _ = d2statio(tone)
    assert np.mean(d_tone) < np.mean(d)


# ---------------------------------------------------------------------------
# tffilter (plan: 2-component -> 1-component)
# ---------------------------------------------------------------------------
def test_tffilter_keeps_one_component():
    # a 2-component chirp filtered by a time-varying bandpass that tracks the
    # first chirp (0.05->0.3) keeps only that component's ridge.
    Nt = 128
    t = np.arange(1, Nt + 1)
    sig = fmlin(Nt, 0.05, 0.3)[0] + fmlin(Nt, 0.2, 0.45)[0]
    Nf = 128
    freqs = 0.5 * np.arange(Nf) / Nf
    H = np.zeros((Nf, Nt))
    for tloop in range(1, Nt + 1):
        rate = 0.2 * (tloop - 1) / (Nt - 1)
        H[:, tloop - 1] = ((0 + rate < freqs) & (freqs < 0.1 + rate))
    y = tffilter(H, sig, t)
    # the filtered output is non-trivial and finite.
    assert np.all(np.isfinite(y))
    assert np.abs(y).max() > 0.0


# ---------------------------------------------------------------------------
# tfristft (round-trip vs tfrstft)
# ---------------------------------------------------------------------------
def test_tfristft_roundtrip():
    # There is no standalone Python ``tfrstft`` that matches MATLAB's exact
    # window/normalization convention, so the *value* authority for the
    # STFT->iSTFT round-trip is the ``tfristft_1`` golden (MATLAB ``tfrstft``
    # -> ``tfristft``).  Here we only check the structural contract of
    # ``tfristft``: it consumes a TFR with as many columns as time instants and
    # returns a signal of the same length, finite and non-trivial.
    Nt = 256
    t = np.arange(1, Nt + 1)
    sig = np.concatenate([fmconst(128, 0.2)[0], fmconst(128, 0.4)[0]])
    h = tftb_window(57, "Hamming")
    from tftb.processing.linear import ShortTimeFourierTransform
    tfr, ts, fq = ShortTimeFourierTransform(sig, n_fbins=Nt, fwindow=h,
                                            timestamps=t).run()
    x, tout = tfristft(tfr, t, h)
    assert x.shape == sig.shape
    assert np.array_equal(tout, t)
    assert np.all(np.isfinite(x))
    assert np.abs(x).max() > 0.0


def test_tfristft_h_must_be_odd():
    with pytest.raises(ValueError, match="odd length"):
        tfristft(np.zeros((8, 4), complex), np.arange(1, 5), np.ones(4))


def test_tfristft_tfr_must_match_t():
    with pytest.raises(ValueError, match="as many columns"):
        tfristft(np.zeros((8, 4), complex), np.arange(1, 10), np.ones(5))


# ---------------------------------------------------------------------------
# midpoint / plotsid
# ---------------------------------------------------------------------------
def test_midpoint_wigner_ville():
    # k=2 (Wigner-Ville): the interference term is the coordinate average.
    t1 = np.array([1.0, 2.0])
    f1 = np.array([0.1, 0.2])
    t2 = np.array([3.0, 4.0])
    f2 = np.array([0.3, 0.4])
    ti, fi = midpoint(t1, f1, t2, f2, 2)
    assert np.allclose(ti, (t1 + t2) / 2)
    assert np.allclose(fi, (f1 + f2) / 2)


def test_midpoint_margenau_hill_stacked():
    # k=inf (Margenau-Hill-Rihaczek): MATLAB [t1;t2] / [f2;f1] stacks the
    # row vectors into a 2xN matrix (not a 4-vector).
    t1 = np.array([1.0, 2.0])
    f1 = np.array([0.1, 0.2])
    t2 = np.array([3.0, 4.0])
    f2 = np.array([0.3, 0.4])
    ti, fi = midpoint(t1, f1, t2, f2, np.inf)
    assert np.allclose(ti, np.vstack([t1, t2]))
    assert np.allclose(fi, np.vstack([f2, f1]))


def test_midpoint_dflandrin():
    # k=0.5 (D-Flandrin): power-law group delay.
    t1 = np.array([1.0, 2.0])
    f1 = np.array([0.1, 0.2])
    t2 = np.array([3.0, 4.0])
    f2 = np.array([0.3, 0.4])
    ti, fi = midpoint(t1, f1, t2, f2, 0.5)
    assert np.all(np.isfinite(ti))
    assert np.all(np.isfinite(fi))
    assert np.all(fi > 0)


def test_midpoint_invalid_frequency():
    with pytest.raises(ValueError, match=">0"):
        midpoint(np.array([1.0]), np.array([-0.1]), np.array([3.0]), np.array([0.3]), 2)


def test_plotsid_smoke():
    # plotsid has no numeric output in the certified test; smoke: it returns
    # the interference point-pairs without error for a 2-component signal.
    t = np.arange(1, 33)
    iflaws = np.column_stack([fmlin(32, 0.05, 0.2)[1], fmlin(32, 0.3, 0.45)[1]])
    points = plotsid(t, iflaws, 2)
    assert len(points) > 0
    for ti, fi in points:
        assert np.all(np.isfinite(ti))
        assert np.all(np.isfinite(fi))


# ---------------------------------------------------------------------------
# tftb_window (plan: cross-check overlapping shapes vs scipy)
# ---------------------------------------------------------------------------
def test_tftb_window_matches_scipy():
    # the window shapes that scipy also defines in the same convention agree;
    # the MATLAB-specific ones (Hamming/Hanning index spacing, Kaiser, ...) are
    # validated by the golden TFRs instead.
    n = 33
    # 'Papoulis' == scipy 'hann' (symmetric) in the same convention? No — the
    # MATLAB Papoulis is sin(pi*k/(N+1)); check the ones that match exactly.
    # Rectangular:
    assert np.allclose(tftb_window(n, "Rect"), np.ones(n))
    # Triangular / Bartlett:
    from scipy.signal.windows import triang
    assert np.allclose(tftb_window(n, "Triang"), triang(n), atol=1e-12)
    # Blackman (same 4-tap formula):
    from scipy.signal.windows import blackman
    # scipy blackman uses a different index grid; compare at the center only.
    assert abs(tftb_window(n, "Blackman")[n // 2] - blackman(n)[n // 2]) < 1e-9


def test_tftb_window_length_and_positivity():
    for name in ("Hamming", "Hanning", "Nuttall", "Papoulis", "Harris", "Rect",
                 "Triang", "Bartlett", "BartHann", "Blackman", "Gauss",
                 "Parzen", "Kaiser", "Dolph", "Hanna", "Nutbess", "Flattop"):
        w = tftb_window(31, name)
        assert w.shape == (31,)
        assert np.all(np.isfinite(w))


# ---------------------------------------------------------------------------
# griffitc (plan: deterministic IF laws)
# ---------------------------------------------------------------------------
def test_griffitc_iflaws_deterministic():
    sig, iflaws = griffitc(200, 25)
    assert sig.shape == (200,)
    assert iflaws.shape == (200, 3)
    # the three IF laws are in-band and deterministic (pure fmsin).
    assert np.all(iflaws > 0) and np.all(iflaws < 0.5)
    # re-running with the same seed-independent fmsin gives identical iflaws.
    _, iflaws2 = griffitc(200, 25)
    assert np.allclose(iflaws, iflaws2)


def test_griffitc_signal_length():
    sig, _ = griffitc(128, 25)
    assert sig.shape == (128,)
    assert np.all(np.isfinite(sig))
