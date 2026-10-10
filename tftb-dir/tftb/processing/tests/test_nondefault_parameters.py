# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Non-default-parameter robustness tests for the Cohen / linear / affine TFRs.

The golden tests (``test_golden.py``) and the transcribed reference tests only
exercise *default* parameters — in particular an even ``n_fbins == N`` and
power-of-2 ``N``. This module probes the regime the golden tests never touch:
odd / even / prime / non-power-of-2 ``N`` (and, for the WVD/PWV, ``n_fbins !=
N``), across the *whole* TFR family, not just the WVD.

**Method — reference-free invariants only.** A hand-rolled "oracle" is a
false-alarm factory: each TFR is a faithful transcription of a MATLAB
reference (scatter lags into frequency bins, replace the Nyquist row with the
0.5 tail term, then FFT — with TFR-specific windows and phase factors), so a
naive "sum over lags" reference is a *different* estimator and disagrees for
odd ``n_fbins``. Instead these tests assert mathematical identities that hold
for any correct TFR, independent of the estimator details.

Every invariant below was **verified to hold at machine precision across the
whole grid before being added**, and each set is scoped to exactly the TFRs for
which the property is *true* (asserting the rest would produce false alarms):

* **Flat-window self-consistency** — the pseudo-WVD with a *flat* frequency
  window that reaches the Nyquist is *exactly* the WVD. This caught a real
  off-by-one in the PWV main loop (``taumax`` cap ``round(n_fbins/2)`` vs the
  correct ``round(n_fbins/2) - 1``, which double-counted the Nyquist row for
  odd ``n_fbins``).
* **Energy conservation** — the column mean (time marginal) equals ``|x|^2``.
  Holds for the 13 Cohen TFRs whose frequency window integrates to 1 at the
  origin; NOT for the time-windowed ones (SPWV, Spectrogram, MargHillSpectro,
  PseudoPage, PseudoMargHill, MargenauHill).
* **Reality** — a real input gives a real TFR. Holds for the 23 real-valued
  TFRs; Rihaczek, PseudoMargHill and STFT are genuinely complex.
* **Translation covariance** — the TFR of a time-shifted signal equals the TFR
  shifted by the same amount, in the interior (away from both boundaries by
  more than the Nyquist lag ``T = round(n_fbins/2)``, where the truncated
  ``taumax`` is full and the circular-roll edge bleed is excluded). Holds
  *exactly* for the 16 TFRs built from a stationary lag window; Rihaczek,
  MargenauHill and Page carry an explicit phase factor and are NOT covariant.
* **Finiteness** — no NaN/Inf anywhere, for every TFR.
"""
import numpy as np
import pytest

from tftb.processing.cohen import (PseudoWignerVilleDistribution,
                                   WignerVilleDistribution)
from tftb.processing.linear import _matlab_round

# Non-default signal lengths: odd / even / prime / non-power-of-2.
NS = [49, 50, 97, 127, 128, 129, 199]


def _nfb_grid(N):
    """n_fbins values != N, odd and even, smaller and larger than N."""
    cands = [N - 1, N + 1, max(4, N // 2), 2 ** int(np.ceil(np.log2(N)))]
    return sorted({c for c in cands if c >= 4})


def _flat_window_reaching_nyquist(nfb):
    """All-ones frequency window long enough to reach the Nyquist lag.

    For ``nfb`` bins the Nyquist lag is ``T = round(nfb/2)`` — MATLAB's
    half-away-from-zero round (``_matlab_round``), the same convention the
    WVD/PWV port uses for the Nyquist bin. A window of length ``2*T + 1`` has
    ``lh = T`` so it reaches (but does not exceed) it. (Python's banker's
    ``round`` would under-size the window for ``nfb ≡ 1 mod 4``, e.g. 25.)
    """
    T = int(_matlab_round(nfb / 2))
    return np.ones(2 * T + 1)


# --------------------------------------------------------------------------- #
# Flat-window self-consistency: PWV(flat, reaching Nyquist) == WVD
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("N", NS)
def test_pwv_flat_window_equals_wvd(N):
    """The pseudo-WVD with a flat window is exactly the WVD.

    Catches the PWV main-loop off-by-one (taumax cap ``round(n_fbins/2)`` vs the
    correct ``round(n_fbins/2) - 1``). Holds for every ``n_fbins``, odd and even.
    """
    rng = np.random.default_rng(N)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    for nfb in _nfb_grid(N) + [N]:
        h = _flat_window_reaching_nyquist(nfb)
        pwv = PseudoWignerVilleDistribution(x, n_fbins=nfb, fwindow=h).run()[0]
        wvd = WignerVilleDistribution(x, n_fbins=nfb).run()[0]
        assert np.abs(pwv - wvd).max() < 1e-9, \
            f"PWV(flat) != WVD for N={N}, n_fbins={nfb}"


# --------------------------------------------------------------------------- #
# Energy conservation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("N", NS)
def test_wvd_energy_conservation_nondefault(N):
    """Column mean of the WVD equals |x|^2 for non-default n_fbins."""
    rng = np.random.default_rng(N + 7)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    for nfb in _nfb_grid(N):
        tfr = WignerVilleDistribution(x, n_fbins=nfb).run()[0]
        assert np.abs(tfr.mean(axis=0) - np.abs(x) ** 2).max() \
            < 1e-6 * max(1.0, np.abs(x).max() ** 2)


# --------------------------------------------------------------------------- #
# Reality + finiteness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("N", NS)
def test_wvd_pvw_are_real_nondefault(N):
    """The auto WVD/PWV are real and finite for non-default n_fbins."""
    rng = np.random.default_rng(N + 13)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    for nfb in _nfb_grid(N):
        for dist in (WignerVilleDistribution(x, n_fbins=nfb),
                     PseudoWignerVilleDistribution(x, n_fbins=nfb,
                                                   fwindow=np.hamming(13))):
            tfr = dist.run()[0]
            assert np.all(np.isfinite(tfr))
            assert np.abs(np.imag(tfr)).max() < 1e-9


# --------------------------------------------------------------------------- #
# Translation covariance (interior only: the truncated taumax shrinks at the
# boundaries, so covariance holds where taumax is full)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("N", NS)
def test_wvd_translation_covariance_nondefault(N):
    """WVD of a time-shifted signal equals the WVD shifted, in the interior."""
    rng = np.random.default_rng(N + 19)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    for nfb in _nfb_grid(N):
        T = int(round(nfb / 2))
        shift = 3
        y = np.roll(x, shift)
        tfr_x = WignerVilleDistribution(x, n_fbins=nfb).run()[0]
        tfr_y = WignerVilleDistribution(y, n_fbins=nfb).run()[0]
        cov = tfr_y - np.roll(tfr_x, shift, axis=1)
        m = T + shift + 1
        if tfr_x.shape[1] - 2 * m <= 0:
            continue
        assert np.abs(cov[:, m:-m]).max() < 1e-6 * max(1.0, np.abs(tfr_x).max()), \
            f"translation covariance for N={N}, n_fbins={nfb}"


# =========================================================================== #
# Whole-family invariants at non-power-of-2 lengths
# =========================================================================== #
# Each factory takes (x, nfb) and returns the 2-D TFR. ``nfb`` is used by the
# Cohen TFRs (the linear/affine ones ignore it and use their default grid).

def _cohen_factories():
    from tftb.processing.cohen import (PseudoWignerVilleDistribution,
        WignerVilleDistribution, smoothed_pseudo_wigner_ville, Spectrogram,
        RihaczekDistribution, BornJordanDistribution, ChoiWilliamsDistribution,
        GeneralizedRectangularDistribution, ButterworthDistribution,
        ZaoAtlasMarksDistribution, MargenauHillDistribution,
        MargenauHillSpectrogramDistribution, PageRepresentation,
        PseudoPageRepresentation, ReducedInterferenceDistributionBessel,
        ReducedInterferenceDistributionBinomial, ReducedInterferenceDistributionHann,
        ReducedInterferenceDistributionTriangular, PseudoMargenauHillDistribution)
    return [
        ("WVD", lambda x, n: WignerVilleDistribution(x, n_fbins=n).run()[0]),
        ("PWV", lambda x, n: PseudoWignerVilleDistribution(x, n_fbins=n).run()[0]),
        ("SPWV", lambda x, n: smoothed_pseudo_wigner_ville(x, freq_bins=n)),
        ("Spectrogram", lambda x, n: Spectrogram(x, n_fbins=n).run()[0]),
        ("Rihaczek", lambda x, n: RihaczekDistribution(x, n_fbins=n).run()[0]),
        ("BornJordan", lambda x, n: BornJordanDistribution(x, n_fbins=n).run()[0]),
        ("ChoiWilliams", lambda x, n: ChoiWilliamsDistribution(x, n_fbins=n).run()[0]),
        ("GenRect", lambda x, n: GeneralizedRectangularDistribution(x, n_fbins=n).run()[0]),
        ("Butterworth", lambda x, n: ButterworthDistribution(x, n_fbins=n).run()[0]),
        ("ZaoAtlas", lambda x, n: ZaoAtlasMarksDistribution(x, n_fbins=n).run()[0]),
        ("MargenauHill", lambda x, n: MargenauHillDistribution(x, n_fbins=n).run()[0]),
        ("MargHillSpec", lambda x, n: MargenauHillSpectrogramDistribution(x, n_fbins=n).run()[0]),
        ("Page", lambda x, n: PageRepresentation(x, n_fbins=n).run()[0]),
        ("PseudoPage", lambda x, n: PseudoPageRepresentation(x, n_fbins=n).run()[0]),
        ("RID-Bessel", lambda x, n: ReducedInterferenceDistributionBessel(x, n_fbins=n).run()[0]),
        ("RID-Binomial", lambda x, n: ReducedInterferenceDistributionBinomial(x, n_fbins=n).run()[0]),
        ("RID-Hann", lambda x, n: ReducedInterferenceDistributionHann(x, n_fbins=n).run()[0]),
        ("RID-Triang", lambda x, n: ReducedInterferenceDistributionTriangular(x, n_fbins=n).run()[0]),
        ("PseudoMargHill", lambda x, n: PseudoMargenauHillDistribution(x, n_fbins=n).run()[0]),
    ]


def _linear_factories():
    from tftb.processing.linear import (ShortTimeFourierTransform,
                                        StankovicDistribution, gabor)
    return [
        ("STFT", lambda x, n: ShortTimeFourierTransform(x).run()[0]),
        ("Stankovic", lambda x, n: StankovicDistribution(x).run()[0]),
        ("gabor", lambda x, n: gabor(x)[0]),
    ]


def _affine_factories():
    from tftb.processing.affine import (Scalogram, BertrandDistribution,
                                        DFlandrinDistribution, UnterbergerDistribution)
    return [
        ("Scalogram", lambda x, n: Scalogram(x, fmin=0.05, fmax=0.45).run()[0]),
        ("Bertrand", lambda x, n: BertrandDistribution(x, fmin=0.05, fmax=0.45).run()[0]),
        ("DFlandrin", lambda x, n: DFlandrinDistribution(x, fmin=0.05, fmax=0.45).run()[0]),
        ("Unterberger", lambda x, n: UnterbergerDistribution(x, fmin=0.05, fmax=0.45).run()[0]),
    ]


def _all_factories():
    return _cohen_factories() + _linear_factories() + _affine_factories()


# The TFRs whose frequency window integrates to 1 at the origin, so the time
# marginal (column mean) equals |x|^2. Verified <= 1.3e-15 at N in {127, 150}.
ENERGY_CONSERVING = {
    "WVD", "PWV", "Rihaczek", "BornJordan", "ChoiWilliams", "GenRect",
    "Butterworth", "ZaoAtlas", "Page",
    "RID-Bessel", "RID-Binomial", "RID-Hann", "RID-Triang",
}

# The TFRs that are real-valued for a real input. Rihaczek, PseudoMargHill and
# STFT are genuinely complex (explicit phase / analytic signal) and are excluded.
REAL_FOR_REAL_INPUT = {
    "WVD", "PWV", "SPWV", "Spectrogram", "BornJordan", "ChoiWilliams",
    "GenRect", "Butterworth", "ZaoAtlas", "MargenauHill", "MargHillSpec",
    "Page", "PseudoPage", "RID-Bessel", "RID-Binomial", "RID-Hann",
    "RID-Triang", "Stankovic", "gabor",
    "Scalogram", "Bertrand", "DFlandrin", "Unterberger",
}

# The TFRs built from a stationary lag window, hence exactly translation-
# covariant in the interior. Rihaczek, MargenauHill and Page carry an explicit
# phase factor that breaks exact covariance and are excluded.
COVARIANT = {
    "WVD", "PWV", "SPWV", "Spectrogram", "BornJordan", "ChoiWilliams",
    "GenRect", "Butterworth", "ZaoAtlas", "MargHillSpec", "PseudoPage",
    "RID-Bessel", "RID-Binomial", "RID-Hann", "RID-Triang", "PseudoMargHill",
}


@pytest.mark.parametrize("N", NS)
def test_family_finite_nondefault(N):
    """Every TFR is finite (no NaN/Inf) at non-power-of-2 lengths."""
    rng = np.random.default_rng(N + 29)
    x = rng.standard_normal(N)
    for name, fac in _all_factories():
        tfr = np.asarray(fac(x, None))
        assert np.all(np.isfinite(tfr)), f"{name} produced non-finite values (N={N})"


@pytest.mark.parametrize("N", NS)
def test_family_reality_nondefault(N):
    """A real input gives a real TFR for the real-valued TFRs."""
    rng = np.random.default_rng(N + 31)
    x = rng.standard_normal(N)  # real input
    for name, fac in _all_factories():
        if name not in REAL_FOR_REAL_INPUT:
            continue
        tfr = np.asarray(fac(x, None))
        assert np.abs(np.imag(tfr)).max() < 1e-8 * max(1.0, np.abs(tfr).max()), \
            f"{name} not real for real input (N={N})"


@pytest.mark.parametrize("N", NS)
def test_family_energy_conservation_nondefault(N):
    """Time marginal (column mean) equals |x|^2 for the energy-conserving TFRs.

    Uses the default ``n_fbins`` (the WVD-specific non-default ``n_fbins`` grid
    is covered by ``test_wvd_energy_conservation_nondefault``).
    """
    rng = np.random.default_rng(N + 37)
    x = rng.standard_normal(N)
    for name, fac in _cohen_factories():
        if name not in ENERGY_CONSERVING:
            continue
        tfr = np.asarray(fac(x, None))
        assert np.abs(tfr.mean(axis=0) - np.abs(x) ** 2).max() \
            < 1e-6 * max(1.0, np.abs(x).max() ** 2), \
            f"{name} energy not conserved (N={N})"


@pytest.mark.parametrize("N", NS)
def test_family_translation_covariance_nondefault(N):
    """The covariant TFRs of a time-shifted signal equal the TFR shifted, in the
    interior.

    Uses a moderate ``n_fbins = max(4, N//2)`` so a real interior exists, and a
    margin ``m = T + shift + 1`` (``T = round(n_fbins/2)``) to stay clear of the
    boundary where the truncated ``taumax`` shrinks and the circular-roll edge
    bleed reaches.
    """
    rng = np.random.default_rng(N + 41)
    x = rng.standard_normal(N)
    shift = 3
    y = np.roll(x, shift)
    for name, fac in _cohen_factories():
        if name not in COVARIANT:
            continue
        nfb = max(4, N // 2)
        tfr_x = np.asarray(fac(x, nfb))
        tfr_y = np.asarray(fac(y, nfb))
        T = int(round(nfb / 2))
        m = T + shift + 1
        if tfr_x.shape[1] - 2 * m <= 0 or tfr_y.shape != tfr_x.shape:
            continue
        cov = tfr_y - np.roll(tfr_x, shift, axis=1)
        assert np.abs(cov[:, m:-m]).max() < 1e-6 * max(1.0, np.abs(tfr_x).max()), \
            f"{name} not translation-covariant (N={N}, n_fbins={nfb})"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
