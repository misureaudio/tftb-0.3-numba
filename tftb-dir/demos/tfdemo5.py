# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""tfdemo5 — Affine class time-frequency distributions.

Port of ``tfdemo5.m`` (O. Lemoine, July 1996, CNRS).

The affine class gathers the quadratic TFRs covariant by translation in time
and dilation (the WVD belongs to it once an arbitrary non-zero frequency
nu0 is introduced and the scale a = nu0/nu). Sections:

1. The **scalogram** (Morlet wavelet) of two gaussian atoms — the affine
   counterpart of the spectrogram, with the same time/frequency-resolution
   tradeoff.
2. (Skipped) the ASPWVD scalogram->WVD movie — needs ``tfrspaw``
   (the affine smoothed pseudo WVD), not yet ported.
3. The **Bertrand** distribution localizing a hyperbolic group-delay signal
   (``gdpower(128, k=0)``).
4. The **D-Flandrin** distribution localizing a ``1/sqrt(nu)`` group-delay
   signal (``gdpower(128, k=1/2)``).
5. The **active Unterberger** distribution localizing a ``1/nu^2``
   group-delay signal (``gdpower(128, k=-1)``).
6. (Skipped) the wide-band ambiguity function of an Altes signal — needs
   ``ambifuwb``, not yet ported.
7. The **interference structure** of the affine Wigner distributions for a
   sinusoidal-FM signal (``plotsid``) for k = 0 (Bertrand), 2 (WVD), -1
   (active Unterberger).
8. The WVD and **Bertrand** distributions of a bat echolocation signal
   (the pseudo-Bertrand comparison needs the unported ``tfrspaw``).

Usage::

    python demos/tfdemo5.py [--save <dir>] [--no-pause]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import hilbert
from tftb.generators import atoms, fmsin, gdpower
from tftb.processing.affine import (BertrandDistribution,
                                    DFlandrinDistribution, Scalogram,
                                    UnterbergerDistribution)
from tftb.processing.analysis import plotsid
from tftb.processing.cohen import WignerVilleDistribution

from demos._common import (init, note, pause, save_or_show, section,
                           show_tfr)


def _plot_sid(t, ifl, k, name):
    """Plot the IF law with the affine-Wigner interference structure for
    group-delay exponent ``k`` (port of ``plotsid.m`` + ``plotifl.m``)."""
    ifl2d = np.asarray(ifl).reshape(-1, 1)          # one component
    pts = plotsid(t, ifl2d, k)
    allt = np.concatenate([a for a, _ in pts])
    allf = np.concatenate([b for _, b in pts])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.asarray(t), np.asarray(ifl).ravel(), "k-", lw=1.5,
            label="IF law")
    ax.scatter(allt, allf, s=3, c="r", alpha=0.35, label="interference pts")
    ax.set_xlabel("Time")
    ax.set_ylabel("Instantaneous frequency")
    ax.set_title(f"Interference structure (k={k}) — {name}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 0.5)
    save_or_show(fig, name)


def main():
    init()

    # ------------------------------------------------------------------ #
    # 1. The scalogram
    # ------------------------------------------------------------------ #
    section("1. Scalogram (Morlet wavelet) of two gaussian atoms")
    sig = atoms(128, np.array([[38, 0.1, 32, 1], [96, 0.35, 32, 1]]))
    tfr, t, f, _wt = Scalogram(sig, fmin=0.01, fmax=0.22).run()
    show_tfr(tfr, t, f, "Scalogram of two atoms (time/frequency resolution "
             "tradeoff)", cmap="inferno")
    pause()

    # ------------------------------------------------------------------ #
    # 2. ASPWVD scalogram -> WVD movie — SKIPPED
    # ------------------------------------------------------------------ #
    note("Section 2 (ASPWVD scalogram->WVD movie) is skipped: it needs "
         "tfrspaw (the affine smoothed pseudo WVD), which is not yet ported.")
    pause()

    # ------------------------------------------------------------------ #
    # 3. Bertrand distribution
    # ------------------------------------------------------------------ #
    section("3. Bertrand distribution (hyperbolic group delay, k=0)")
    sig = gdpower(128, 0)[0]
    tfr, t, f = BertrandDistribution(sig, fmin=0.01, fmax=0.22,
                                     n_voices=128).run()
    show_tfr(tfr, t, f, "Bertrand distribution of gdpower(k=0) "
             "(localized on the hyperbolic group delay)", cmap="bwr")
    pause()

    # ------------------------------------------------------------------ #
    # 4. D-Flandrin distribution
    # ------------------------------------------------------------------ #
    section("4. D-Flandrin distribution (1/sqrt(nu) group delay, k=1/2)")
    sig = gdpower(128, 0.5)[0]
    tfr, t, f = DFlandrinDistribution(sig, fmin=0.01, fmax=0.22,
                                      n_voices=128).run()
    show_tfr(tfr, t, f, "D-Flandrin distribution of gdpower(k=1/2) "
             "(localized on the 1/sqrt(nu) group delay)", cmap="bwr")
    pause()

    # ------------------------------------------------------------------ #
    # 5. Active Unterberger distribution
    # ------------------------------------------------------------------ #
    section("5. Active Unterberger distribution (1/nu^2 group delay, k=-1)")
    sig = gdpower(128, -1)[0]
    tfr, t, f = UnterbergerDistribution(sig, form="A", fmin=0.01, fmax=0.22,
                                        n_voices=172).run()
    show_tfr(tfr, t, f, "Active Unterberger distribution of gdpower(k=-1) "
             "(localized on the 1/nu^2 group delay)", cmap="bwr")
    pause()

    # ------------------------------------------------------------------ #
    # 6. Wide-band ambiguity function — SKIPPED
    # ------------------------------------------------------------------ #
    note("Section 6 (wide-band ambiguity function of an Altes signal) is "
         "skipped: it needs ambifuwb, which is not yet ported.")
    pause()

    # ------------------------------------------------------------------ #
    # 7. Interference structure (plotsid)
    # ------------------------------------------------------------------ #
    section("7. Interference structure of the affine Wigner distributions")
    sig, ifl = fmsin(128)
    t = np.arange(1, 129)
    _plot_sid(t, ifl, 0, "sid_bert")
    pause()
    _plot_sid(t, ifl, 2, "sid_wv")
    pause()
    _plot_sid(t, ifl, -1, "sid_unter")
    pause()

    # ------------------------------------------------------------------ #
    # 8. Bat echolocation signal: WVD vs Bertrand
    # ------------------------------------------------------------------ #
    section("8. Bat echolocation signal: WVD and Bertrand")
    bat_path = os.path.join("D:", os.sep, "TOOLBOX", "tftb-0.3", "data", "bat.mat")
    bat = np.asarray(loadmat(bat_path, squeeze_me=True)["bat"]).ravel()
    N = 1648
    sig = hilbert(bat[400:(400 + N)])
    # WVD at all times (the port's WVD run() indexes the signal by the output
    # column, so subsampled timestamps are not supported — the MATLAB demo's
    # 1:8:N subsampling is only to keep the plot light; the distribution is
    # the same, sampled more densely here).
    tfr, t, f = WignerVilleDistribution(sig, n_fbins=256).run()
    show_tfr(tfr, t, f, "WVD of the bat signal (interference terms from the "
             "nonlinear FM)", cmap="bwr")
    tfr, t, f = BertrandDistribution(sig, fmin=0.1, fmax=0.4,
                                     n_voices=256).run()
    show_tfr(tfr, t, f, "Bertrand distribution of the bat signal (better "
             "localization; residual interference)", cmap="bwr")
    note("The MATLAB demo's second plot of this section is the pseudo-"
         "Bertrand distribution (tfrspaw), which cancels the residual "
         "interference; it is not yet ported.")
    pause()

    print("\nEnd of tfdemo5.")


if __name__ == "__main__":
    main()
