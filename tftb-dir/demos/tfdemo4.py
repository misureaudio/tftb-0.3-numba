# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""tfdemo4 — Cohen's class time-frequency distributions.

Port of ``tfdemo4.m`` (O. Lemoine, May 1996, CNRS).

Demonstrates:

1. The Wigner-Ville distribution (WVD) of a linear chirp (near-perfect
   localization, negative values) and of a Doppler signal (interference
   terms from the WVD's bilinearity).
2. The interference geometry of the WVD (movie: two atoms whose separation
   increases then decreases — the interference period/direction change).
3. The pseudo-WVD (PWV) of four atoms: the oscillating interferences are
   attenuated, at the cost of a spreading of the signal terms in frequency.
4. The importance of the analytic signal: the WVD of the *real* part of a
   two-atom signal shows four terms (aliasing) plus extra interference; the
   analytic form shows only two.
5. The smoothed pseudo-WVD (SPWV): WVD vs PWV vs SPWV of a two-component
   signal, and the SPWV→WVD transition (movie: a 4-atom signal with varying
   time/frequency smoothing).
6. The narrow-band ambiguity function: AF-signal terms near the origin,
   AF-interference terms away from it.
7. The Rihaczek / Margenau-Hill distributions (interference geometry
   different from the WVD's) and the Choi-Williams distribution (movie: four
   atoms rotating around the center of the time-frequency plane).
8. (Skipped) the ``paramfun`` weighting-function comparison — needs
   ``paramfun.mat``, which is absent (the MATLAB demo skips it too).

Usage::

    python demos/tfdemo4.py [--save <dir>] [--no-pause]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib.pyplot as plt
import numpy as np
from tftb.generators import amgauss, atoms, doppler, fmconst, fmlin, sigmerge
from tftb.processing.ambiguity import narrow_band
from tftb.processing.cohen import (ChoiWilliamsDistribution,
                                   MargenauHillDistribution,
                                   PseudoWignerVilleDistribution,
                                   Spectrogram, WignerVilleDistribution,
                                   smoothed_pseudo_wigner_ville)
from tftb.processing.utils import tftb_window

from demos._common import (init, note, pause, run_frames, save_or_show, section,
                           shift_tfr, show_tfr)
from demos._common import sig as _sig


# --------------------------------------------------------------------------- #
# Frequency-axis conventions (verified against the port)
# --------------------------------------------------------------------------- #
# WVD / CW / SPWV return rows in fft order with f[i] = i/(2N) in [0, 0.5):
#   NO shift.  PWV / MH / RI / Spectrogram return a wrapped axis: shift_tfr.
def _wvd_axis(n):
    return 0.5 * np.arange(n) / n


# --------------------------------------------------------------------------- #
# Movie helpers (ports of movwv2at.m / movsp2wv.m / movcw4at.m)
# --------------------------------------------------------------------------- #
def _odd(x):
    """MATLAB ``odd``: floor, and if even -> ceil (the next odd integer)."""
    x = np.asarray(x, float)
    y = np.floor(x)
    y = np.where(y % 2 == 0, np.ceil(x), y)
    y = np.where(y % 2 == 0, y + 1, y)
    return y


def _movie_wv2at(N, Np=15):
    """Port of movwv2at.m: WVD of two atoms with increasing/decreasing
    frequency separation. Returns (frames, t, f_list)."""
    Np = int(_odd(Np))
    c = Np / 0.2
    ampl = amgauss(N // 2, N // 2, N // 2)
    t = np.arange(N)
    frames, f_list = [], []
    half = (Np + 1) // 2
    for k in range(1, half + 1):
        s = np.concatenate([
            ampl * _sig(fmconst(N // 2, 0.25 - (k - 1) / c)),
            ampl * _sig(fmconst(N // 2, 0.25 + (k - 1) / c)),
        ])
        tfr, _, f = WignerVilleDistribution(s).run()
        frames.append(tfr)
        f_list.append(f)
    # Mirror for the decreasing-separation half.
    for k in range(half + 1, Np + 1):
        frames.append(frames[Np - k])
        f_list.append(f_list[Np - k])
    return frames, t, f_list


def _movie_sp2wv(N, Np=8):
    """Port of movsp2wv.m: a 4-atom signal, transitioning spectrogram ->
    SPWV (varying time/frequency smoothing) -> WVD. Returns (frames, t,
    f_list)."""
    Np = int(_odd(Np)) - 1
    t1, t2, t3, t4 = N / 4, 3 * N / 4, N / 4, 3 * N / 4
    f1, f2, f3, f4 = 0.125, 0.125, 0.375, 0.375
    T = np.sqrt(2 * N)
    # 1-based MATLAB t0 -> 0-based port t0 - 1
    sig = (amgauss(N, t1, T) * _sig(fmconst(N, f1, t1 - 1))
           + amgauss(N, t2, T) * _sig(fmconst(N, f2, t2 - 1))
           + amgauss(N, t3, T) * _sig(fmconst(N, f3, t3 - 1))
           + amgauss(N, t4, T) * _sig(fmconst(N, f4, t4 - 1)))
    t = np.arange(N)
    frames, f_list = [], []

    # Frame 1: spectrogram (hanning-49), positive-frequency half.
    tfr, _, f = Spectrogram(sig, n_fbins=N, fwindow=tftb_window(49, "gauss")).run()
    tfr, f = shift_tfr(tfr, f)
    frames.append(tfr[:N // 2, :])
    f_list.append(f[:N // 2])

    # Middle frames: SPWV with increasing time/frequency smoothing.
    Ng0 = _odd(np.linspace(3, N / 4, Np // 2 - 1)[::-1])
    Nh0 = _odd(np.linspace(N / 2, 9 * N / 10, Np // 2 - 1))
    for k in range(2, Np // 2 + 1):
        g = tftb_window(int(Ng0[k - 2]), "gauss")
        h = tftb_window(int(Nh0[k - 2]), "gauss")
        tfr = smoothed_pseudo_wigner_ville(
            sig, freq_bins=N, twindow=g, fwindow=h)
        frames.append(tfr)
        f_list.append(_wvd_axis(N))

    # WVD frame.
    tfr, _, f = WignerVilleDistribution(sig).run()
    frames.append(tfr)
    f_list.append(f)

    # Mirror for the second half.
    for k in range(Np // 2 + 2, Np + 1):
        frames.append(frames[Np + 1 - k])
        f_list.append(f_list[Np + 1 - k])
    return frames, t, f_list


def _movie_cw4at(N, Np=7):
    """Port of movcw4at.m: Choi-Williams distribution of four atoms rotating
    around the center of the time-frequency plane. Returns (frames, t,
    f_list)."""
    rho = N / 3
    T = np.sqrt(N)
    t = np.arange(N)
    frames, f_list = [], []
    for k in range(1, Np + 1):
        theta = (k - 1) * np.pi / (2 * (Np - 1))
        s = 0
        for j in range(4):
            th = theta + j * np.pi / 2
            t0 = N / 2 + rho * np.cos(th)
            f0 = 0.5 * (N / 2 + rho * np.sin(th)) / N
            s = s + amgauss(N, t0, T) * _sig(fmconst(N, f0, round(t0) - 1))
        tfr, _, f = ChoiWilliamsDistribution(
            s, n_fbins=N, twindow=tftb_window(N // 2 + 1),
            fwindow=tftb_window(N + 1), sigma=1.0).run()
        frames.append(tfr)
        f_list.append(f)
    return frames, t, f_list


# --------------------------------------------------------------------------- #
def main():
    init()

    # ------------------------------------------------------------------ #
    # 1. WVD of a linear chirp and of a Doppler signal
    # ------------------------------------------------------------------ #
    section("1. Wigner-Ville distribution of a linear chirp and a Doppler signal")
    sig1 = _sig(fmlin(128))
    tfr, t, f = WignerVilleDistribution(sig1).run()
    show_tfr(tfr, t, f, "WVD of a linear chirp (near-perfect localization)",
             cmap="bwr")
    fm3, am3, _ = doppler(128, 50, 13, 10, 200)
    sigd = am3 * fm3
    tfr, t, f = WignerVilleDistribution(sigd).run()
    show_tfr(tfr, t, f, "WVD of a Doppler signal (interference terms visible)",
             cmap="bwr")
    pause()

    # ------------------------------------------------------------------ #
    # 2. Interference geometry of the WVD (movie)
    # ------------------------------------------------------------------ #
    section("2. Interference geometry of the WVD (two atoms, varying separation)")
    frames, t, f_list = _movie_wv2at(128, 15)
    run_frames(frames, t, f_list, "WVD of two atoms (interference geometry)",
               prefix="wv2at")
    pause()

    # ------------------------------------------------------------------ #
    # 3. Pseudo-WVD of four atoms
    # ------------------------------------------------------------------ #
    section("3. Pseudo-WVD of four atoms (interference attenuation)")
    sig = atoms(128, np.array([[32, .15, 20, 1], [96, .15, 20, 1],
                               [32, .35, 20, 1], [96, .35, 20, 1]]))
    tfr, t, f = WignerVilleDistribution(sig).run()
    show_tfr(tfr, t, f, "WVD of four atoms (4 signal + 6 interference terms)",
             cmap="bwr")
    tfr, t, f = PseudoWignerVilleDistribution(sig).run()
    tfr, f = shift_tfr(tfr, f)
    show_tfr(tfr, t, f, "Pseudo-WVD of four atoms (interferences attenuated)")
    pause()

    # ------------------------------------------------------------------ #
    # 4. Importance of the analytic signal
    # ------------------------------------------------------------------ #
    section("4. Importance of the analytic signal (aliasing in the WVD)")
    sig = atoms(128, np.array([[32, 0.15, 20, 1], [96, 0.32, 20, 1]]))
    tfr, t, f = WignerVilleDistribution(np.real(sig)).run()
    show_tfr(tfr, t, f, "WVD of the REAL part (4 terms from aliasing)",
             cmap="bwr")
    tfr, t, f = WignerVilleDistribution(sig).run()
    show_tfr(tfr, t, f, "WVD of the ANALYTIC signal (2 terms, no aliasing)",
             cmap="bwr")
    pause()

    # ------------------------------------------------------------------ #
    # 5. Smoothed pseudo-WVD
    # ------------------------------------------------------------------ #
    section("5. Smoothed pseudo-WVD (WVD vs PWV vs SPWV) + SPWV->WVD movie")
    sig = sigmerge(_sig(fmconst(128, .15)), amgauss(128) * _sig(fmconst(128, 0.4)), 5)
    tfr, t, f = WignerVilleDistribution(sig).run()
    show_tfr(tfr, t, f, "WVD (two terms + interference)", cmap="bwr")
    tfr, t, f = PseudoWignerVilleDistribution(sig).run()
    tfr, f = shift_tfr(tfr, f)
    show_tfr(tfr, t, f, "Pseudo-WVD (frequency smoothing — interference remains)")
    tfr = smoothed_pseudo_wigner_ville(sig)
    show_tfr(tfr, np.arange(128), _wvd_axis(128),
             "Smoothed pseudo-WVD (time smoothing — interference reduced)")
    pause()
    frames, t, f_list = _movie_sp2wv(128, 8)
    run_frames(frames, t, f_list, "Spectrogram -> SPWV -> WVD (smoothing tradeoff)",
               prefix="sp2wv")
    pause()

    # ------------------------------------------------------------------ #
    # 6. Narrow-band ambiguity function
    # ------------------------------------------------------------------ #
    section("6. Narrow-band ambiguity function (two chirps)")
    N = 64
    sig = np.concatenate([
        _sig(fmlin(N, 0.2, 0.5)) * amgauss(N),
        _sig(fmlin(N, 0.3, 0.0)) * amgauss(N),
    ])
    tfr, t, f = WignerVilleDistribution(sig).run()
    show_tfr(tfr, t, f, "WVD of two chirps (interference in the middle)",
             cmap="bwr")
    naf, lag, xi = narrow_band(sig)
    fig, ax = plt.subplots(figsize=(7, 5))
    pc = ax.pcolormesh(lag, xi, np.abs(naf), cmap="inferno", shading="nearest")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Doppler")
    ax.set_title("Narrow-band ambiguity function (signal terms near origin)")
    ax.set_aspect("auto")
    fig.colorbar(pc, ax=ax, label="|AF|")
    save_or_show(fig, "af")
    pause()

    # ------------------------------------------------------------------ #
    # 7. Rihaczek / Margenau-Hill and Choi-Williams
    # ------------------------------------------------------------------ #
    section("7. Margenau-Hill distribution + Choi-Williams movie (4 rotating atoms)")
    sig = atoms(128, np.array([[32, 0.15, 20, 1], [96, 0.32, 20, 1]]))
    tfr, t, f = MargenauHillDistribution(sig).run()
    tfr, f = shift_tfr(tfr, f)
    show_tfr(tfr, t, f, "Margenau-Hill (interference at (t1,f2),(t2,f1))",
             cmap="bwr")
    frames, t, f_list = _movie_cw4at(128)
    run_frames(frames, t, f_list, "Choi-Williams of four rotating atoms",
               prefix="cw4at")
    pause()

    # ------------------------------------------------------------------ #
    # 8. Weighting-function comparison (paramfun) — SKIPPED
    # ------------------------------------------------------------------ #
    note("Section 8 (paramfun weighting-function comparison) is skipped: "
         "it requires paramfun.mat, which is absent from the data directory "
         "(the MATLAB demo skips it too).")

    print("\nEnd of tfdemo4.")


if __name__ == "__main__":
    main()
