# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""End-to-end chain test: atoms -> pWV -> reassignment -> ridges -> Friedman.

The plan's Task 3.2 acceptance: build a 4-atom signal, run the reassigned
pseudo Wigner-Ville, extract the reassignment ridges, and check that the
ridge curves recover the 4 atoms' instantaneous-frequency laws (a closed-form
check — each atom is a constant-frequency Gaussian, so its IF law is the flat
line at ``f0``). Also verifies the Friedman density integrates to 1.

This exercises the full Phase 3 pipeline on real (non-golden) data and is the
integration check that the reassignment core (Task 3.1) and the post-
processing (this task) compose correctly.
"""
import numpy as np
import pytest

from tftb.generators import atoms
from tftb.processing.reassigned import pseudo_wigner_ville
from tftb.processing.postprocessing import ridges, friedman_density

# 4 well-separated constant-frequency Gaussian atoms (t0, f0, spread, amp).
_ATOMS = np.array([
    [32, 0.1, 16, 1],
    [96, 0.2, 16, 1],
    [160, 0.3, 16, 1],
    [224, 0.4, 16, 1],
])
_N = 256


def _chain():
    sig = atoms(_N, _ATOMS)
    t = np.arange(1, _N + 1)
    tfr, rtfr, hat = pseudo_wigner_ville(sig)
    pt, pf = ridges(tfr, hat, t, method='rpwv')
    tifd = friedman_density(tfr, hat)
    return tfr, rtfr, hat, pt, pf, tifd


def test_chain_ridges_recover_if_laws():
    tfr, rtfr, hat, pt, pf, tifd = _chain()
    assert len(pt) > 0, "ridge extraction found no stationary points"
    for t0, f0, spread, _ in _ATOMS:
        # ridge points in the atom's time neighborhood
        m = (pt >= t0 - spread) & (pt <= t0 + spread)
        assert m.sum() > 50, f"too few ridge points near atom t0={t0}"
        # closed-form check: the atom is constant-frequency, so the ridge
        # frequency is the flat line at f0. The median recovers it.
        assert abs(np.median(pf[m]) - f0) < 0.02, (
            f"atom t0={t0}: ridge freq median {np.median(pf[m]):.4f} "
            f"!= IF law {f0}")


def test_chain_clusters_are_distinct():
    tfr, rtfr, hat, pt, pf, tifd = _chain()
    medians = []
    for t0, f0, spread, _ in _ATOMS:
        m = (pt >= t0 - spread) & (pt <= t0 + spread)
        medians.append(np.median(pf[m]))
    medians = np.sort(medians)
    # the 4 recovered IF laws are strictly increasing and well separated
    # (atoms are 0.1 apart in frequency).
    assert (np.diff(medians) > 0.08).all(), (
        f"frequency clusters not distinct: {medians}")
    # and they match the planted frequencies in order
    assert np.allclose(medians, np.sort([a[1] for a in _ATOMS]), atol=0.02)


def test_chain_friedman_unit_integral():
    tfr, rtfr, hat, pt, pf, tifd = _chain()
    assert abs(np.sum(tifd) - 1.0) < 1e-8
    assert not (tifd < 0).any()


def test_chain_rtfr_energy_conserved():
    # the reassigned TFR conserves signal energy (reassignment only moves
    # energy between bins) — a sanity check that the reassignment core
    # (Task 3.1) is wired correctly through the pWV path.
    sig = atoms(_N, _ATOMS)
    tfr, rtfr, hat = pseudo_wigner_ville(sig)
    assert abs(np.vdot(sig, sig).real - np.sum(np.mean(rtfr, axis=0))) < 1e-8
