# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Reference tests for energy-dot extraction (``imextrac`` + ``energy_dots``).

``imextrac`` is golden-validated against MATLAB (``extr_1`` — the docstring
image, exact label layout including the union-by-rename). ``energy_dots``
(the port of ``tfrsurf``) has no MATLAB golden: MATLAB's ``tfrsurf`` passes a
``logical`` ``Binary`` to ``imextrac``, and MATLAB's integer-into-logical
assignment collapses every label to ``true``, so MATLAB's own ``tfrsurf`` hits
its ``error('should never happen')`` branch on any real TFR. The Python port
casts the binary to int (the fix), so it is validated here by a closed-form
2-atom fixture instead: two well-separated constant-frequency Gaussian atoms
must yield exactly two dominant energy dots at the atom locations.
"""
import numpy as np
import pytest

from tftb.generators import amgauss, fmconst
from tftb.processing.cohen import WignerVilleDistribution
from tftb.processing.postprocessing import imextrac, energy_dots

_N = 128
# (t0 1-based, f0) for the two atoms.
_ATOMS = [(32, 0.15), (96, 0.35)]


def _two_atom_tfr():
    sig = (
        amgauss(_N, _ATOMS[0][0], 6) * fmconst(_N, _ATOMS[0][1])
        + amgauss(_N, _ATOMS[1][0], 6) * fmconst(_N, _ATOMS[1][1])
    )
    return WignerVilleDistribution(sig).run()[0]


def test_imextrac_counts_components():
    # a 3x3 image with a single 2x2 block -> one dot (label 2), 4 pixels.
    img = np.zeros((3, 3), dtype=int)
    img[0:2, 0:2] = 1
    labels, nb_dots = imextrac(img)
    assert nb_dots == 2  # background + 1 dot
    assert (labels == 2).sum() == 4
    # two disconnected dots -> nb_dots 3.
    img2 = np.zeros((3, 3), dtype=int)
    img2[0, 0] = img2[2, 2] = 1
    labels2, nb_dots2 = imextrac(img2)
    assert nb_dots2 == 3
    assert set(np.unique(labels2)) == {0, 2, 3}


def test_energy_dots_two_atoms_keep2():
    tfr = _two_atom_tfr()
    tfr2, ordered = energy_dots(tfr, threshold=5, keep=2)
    # exactly 2 dots survive; the top-2 are far larger than the 3rd (the
    # cross-term / interference), so keep=2 isolates the atoms.
    assert len(ordered) >= 3
    assert ordered[0] > 3 * ordered[2] and ordered[1] > 3 * ordered[2], (
        f"atoms not dominant: top3 = {ordered[:3]}")
    # tfr2 keeps exactly the 2 biggest dots' pixels.
    expected_pixels = ordered[0] + ordered[1]
    assert (tfr2 != 0).sum() == expected_pixels
    # everything zeroed outside the kept dots.
    assert (tfr2 == tfr * (tfr2 != 0)).all()


def test_energy_dots_at_atom_locations():
    tfr = _two_atom_tfr()
    tfr2, ordered = energy_dots(tfr, threshold=5, keep=2)
    # the 2 kept dots sit at the atoms' (time, frequency) locations.
    binary = (tfr >= 5 * tfr.max() * 0.01)
    labels, nb_dots = imextrac(binary)
    sizes = np.bincount(labels.ravel(), minlength=nb_dots + 1)
    order = np.argsort(sizes[1:], kind="stable")[::-1]
    centroids = []
    for k in range(2):
        lbl = order[k] + 1
        ys, xs = np.nonzero(labels == lbl)
        centroids.append((xs.mean(), ys.mean()))  # (time, freq)
    # time centroids (0-based) match the atom times (t0 is 1-based).
    times = sorted(c[0] for c in centroids)
    for got, (t0, _) in zip(times, sorted(_ATOMS, key=lambda a: a[0])):
        assert abs(got - (t0 - 1)) < 3.0, (
            f"dot time {got:.1f} != atom t0 {t0} (0-based {t0 - 1})")
    # frequency centroids: the lower-frequency atom is the lower row, and the
    # rows scale linearly with f0 (Wigner-Ville frequency axis).
    freqs = sorted(c[1] for c in centroids)
    f0s = sorted(a[1] for a in _ATOMS)
    assert freqs[0] < freqs[1]
    assert abs(freqs[1] / freqs[0] - f0s[1] / f0s[0]) < 0.1
