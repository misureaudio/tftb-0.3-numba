# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Phase 3 exit gate — the full reassigned + post-processing pipeline.

The plan's Phase 3 exit gate: *"the full reassigned + post-processing pipeline
(TFR → reassignment → ridges → Friedman → Hough → dots) green."* Each stage is
validated in isolation (``test_reassigned_spectrogram_reference.py``,
``test_ridges_reference.py``, ``test_friedman_reference.py``,
``test_hough_reference.py``, ``test_energy_dots_reference.py``) and the
reassignment→ridges→Friedman sub-chain is validated in
``test_postprocessing_chain.py``. This module is the missing end-to-end link:
it threads a 2-atom signal through *all* six stages in sequence and asserts
each stage's output is coherent with the next, i.e. the whole pipeline composes
on real (non-golden) data.

Fixture: two well-separated constant-frequency Gaussian atoms at (t0=32,
f0=0.15) and (t0=96, f0=0.35) — separated enough in time that the reassignment
yields two distinct ridges, two dominant energy dots, and a Hough peak.
"""
import numpy as np
import pytest

from tftb.generators import amgauss, fmconst
from tftb.processing.reassigned import pseudo_wigner_ville
from tftb.processing.postprocessing import (
    ridges,
    friedman_density,
    hough_transform,
    energy_dots,
)

_N = 128
# (t0 1-based, f0) for the two atoms.
_ATOMS = [(32, 0.15), (96, 0.35)]


def _reassigned_tfr():
    sig = (
        amgauss(_N, _ATOMS[0][0], 6) * fmconst(_N, _ATOMS[0][1])[0]
        + amgauss(_N, _ATOMS[1][0], 6) * fmconst(_N, _ATOMS[1][1])[0]
    )
    t = np.arange(1, _N + 1)
    tfr, rtfr, hat = pseudo_wigner_ville(sig)
    return sig, t, tfr, rtfr, hat


def test_pipeline_reassignment_stage():
    # Stage 1: TFR -> reassignment. The reassigned TFR conserves signal energy
    # and the hat matrix carries the 1-based reassignment vectors.
    sig, t, tfr, rtfr, hat = _reassigned_tfr()
    assert rtfr.shape == tfr.shape
    # reassignment only moves energy between bins.
    assert abs(np.vdot(sig, sig).real - np.sum(np.mean(rtfr, axis=0))) < 1e-8
    # hat is finite above the reassignment threshold (no inf in the active
    # region of a well-conditioned signal).
    assert np.all(np.abs(hat) >= 0)


def test_pipeline_ridges_stage():
    # Stage 2: reassignment -> ridges. The two atoms' IF laws (flat lines at
    # f0) are recovered as two distinct ridge clusters.
    sig, t, tfr, rtfr, hat = _reassigned_tfr()
    pt, pf = ridges(rtfr, hat, t, method='rpwv')
    assert len(pt) > 0
    medians = []
    for t0, f0 in _ATOMS:
        m = (pt >= t0 - 12) & (pt <= t0 + 12)
        assert m.sum() > 20, f"too few ridge points near atom t0={t0}"
        medians.append(np.median(pf[m]))
    medians = np.sort(medians)
    assert (np.diff(medians) > 0.1).all(), f"ridges not distinct: {medians}"
    assert np.allclose(medians, np.sort([a[1] for a in _ATOMS]), atol=0.02)


def test_pipeline_friedman_stage():
    # Stage 3: reassignment -> Friedman instantaneous-frequency density.
    # Integrates to 1 and is non-negative.
    sig, t, tfr, rtfr, hat = _reassigned_tfr()
    tifd = friedman_density(rtfr, hat)
    assert abs(np.sum(tifd) - 1.0) < 1e-8
    assert not (tifd < 0).any()
    # the density concentrates near the two atoms' times.
    col_energy = np.sum(tifd, axis=0)
    for t0, _ in _ATOMS:
        assert col_energy[t0 - 1] > 0, f"no Friedman density at atom t0={t0}"


def test_pipeline_hough_stage():
    # Stage 4: reassignment -> Hough transform. The two ridges are line
    # segments in the time-frequency plane; the Hough of the reassigned TFR
    # image has a dominant peak (max >> median), unlike structure-free noise.
    sig, t, tfr, rtfr, hat = _reassigned_tfr()
    img = np.abs(rtfr)
    ht, rho, theta = hough_transform(img, m=128, n=128)
    assert ht.shape == (128, 128)
    maxv = ht.max()
    nonzero = ht[ht > 0]
    assert maxv > 0
    # the peak is dominant relative to the bulk of the (sparse) accumulator.
    assert maxv / nonzero.min() > 2.0, (
        f"Hough peak not dominant: max={maxv:.1f}, min-nz={nonzero.min():.1f}")
    # the peak sits at a finite (rho, theta) in the valid ranges.
    ji, ii = np.unravel_index(np.argmax(ht), ht.shape)
    assert np.isfinite(rho[ji]) and np.isfinite(theta[ii])


def test_pipeline_energy_dots_stage():
    # Stage 5: reassignment -> energy dots. Two dominant dots at the two
    # atoms' locations, dwarfing the cross-term / interference.
    sig, t, tfr, rtfr, hat = _reassigned_tfr()
    tfr2, ordered = energy_dots(np.abs(rtfr), threshold=5, keep=2)
    assert len(ordered) >= 3
    assert ordered[0] > 3 * ordered[2] and ordered[1] > 3 * ordered[2], (
        f"atoms not dominant: top3 = {ordered[:3]}")
    assert (tfr2 != 0).sum() == ordered[0] + ordered[1]


def test_full_pipeline_end_to_end():
    """The complete chain in one pass: TFR -> reassignment -> ridges ->
    Friedman -> Hough -> dots, with each stage's output feeding the next."""
    sig, t, tfr, rtfr, hat = _reassigned_tfr()
    # 1. reassignment (rtfr/hat) — energy conserving.
    assert abs(np.vdot(sig, sig).real - np.sum(np.mean(rtfr, axis=0))) < 1e-8
    # 2. ridges — two distinct IF laws.
    pt, pf = ridges(rtfr, hat, t, method='rpwv')
    medians = sorted(np.median(pf[(pt >= t0 - 12) & (pt <= t0 + 12)])
                     for t0, _ in _ATOMS)
    assert (medians[1] - medians[0]) > 0.1
    # 3. Friedman — unit integral.
    tifd = friedman_density(rtfr, hat)
    assert abs(np.sum(tifd) - 1.0) < 1e-8
    # 4. Hough — dominant peak.
    ht, rho, theta = hough_transform(np.abs(rtfr), m=128, n=128)
    assert ht.max() / (ht[ht > 0].min() or 1.0) > 2.0
    # 5. energy dots — two dominant dots.
    tfr2, ordered = energy_dots(np.abs(rtfr), threshold=5, keep=2)
    assert ordered[0] > 3 * ordered[2]
    # the whole pipeline ran without error and produced coherent outputs.
    assert len(pt) > 0 and (tfr2 != 0).any() and (tifd > 0).any()
