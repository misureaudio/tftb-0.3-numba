# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""tfdemo2 — Non-stationary signals.

Port of ``tfdemo2.m`` (O. Lemoine, May 1996, CNRS).

Demonstrates:

1. Time/frequency localization (``loctime`` / ``locfreq``) of a chirp with
   Gaussian amplitude, and the Heisenberg-Gabor inequality T*B >= 1.
2. The Gaussian signal minimizes the time-bandwidth product.
3. Instantaneous frequency (``inst_freq``) of a linear FM.
4. Group delay (``group_delay``) of the same signal.
5. IF vs group delay: nearly identical for large TB, clearly different
   for small TB.
6. Synthesis of mono-component signals (am*fm, fmconst*amexpo1s,
   ``doppler``, + complex colored noise).
7. Multi-component signals: IF/GD become meaningless; the STFT shows both
   components.

Usage::

    python demos/tfdemo2.py [--save <dir>] [--no-pause]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib.pyplot as plt
import numpy as np
from tftb.generators import (amexpos, amgauss, doppler, fmconst, fmlin,
                             noisecg, sigmerge)
from tftb.processing.freq_domain import group_delay, inst_freq, locfreq
from tftb.processing.linear import ShortTimeFourierTransform
from tftb.processing.plotifl import plotifl
from tftb.processing.time_domain import loctime

from demos._common import (init, pause, save_current, save_or_show, section,
                           shift_tfr, show_tfr, show_time_spectrum)
from demos._common import sig as _sig


def main():
    init()

    # ------------------------------------------------------------------ #
    # 1. Time/frequency localization + Heisenberg-Gabor
    # ------------------------------------------------------------------ #
    section("1. Time and frequency localization (loctime / locfreq)")
    sig = _sig(fmlin(256)) * amgauss(256)
    show_time_spectrum(sig, "Chirp with Gaussian amplitude")
    tm, T = loctime(sig)
    num, B = locfreq(sig)
    print(f"\n  Average time center tm = {tm:.3f}, duration T = {T:.3f}")
    print(f"  Average frequency num = {num:.4f}, bandwidth B = {B:.4f}")
    print(f"  Time-bandwidth product T*B = {T * B:.4f}   (>= 1)")
    pause()

    # ------------------------------------------------------------------ #
    # 2. Gaussian minimizes T*B
    # ------------------------------------------------------------------ #
    section("2. Gaussian signal — minimizes the time-bandwidth product")
    sig = amgauss(256)
    show_time_spectrum(sig, "Gaussian signal")
    tm, T = loctime(sig)
    fm, B = locfreq(sig)
    print(f"\n  T = {T:.4f}, B = {B:.4f}, T*B = {T * B:.4f}")
    print("  (The Gaussian is the most concentrated signal in the")
    print("   time-frequency plane: it attains the Heisenberg-Gabor bound.)")
    pause()

    # ------------------------------------------------------------------ #
    # 3. Instantaneous frequency
    # ------------------------------------------------------------------ #
    section("3. Instantaneous frequency of a linear FM (inst_freq)")
    sig = _sig(fmlin(256))
    ifr, t = inst_freq(sig)
    plotifl(t, ifr, sig[t])
    save_current("ifl")
    pause()

    # ------------------------------------------------------------------ #
    # 4. Group delay
    # ------------------------------------------------------------------ #
    section("4. Group delay of the same signal (group_delay)")
    fnorm = np.arange(0, 0.5 + 1e-9, 0.05)
    gd = group_delay(sig, fnorm)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gd, fnorm)
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized frequency")
    ax.set_title("Group delay estimation")
    ax.grid(True)
    save_or_show(fig, "gd")
    pause()

    # ------------------------------------------------------------------ #
    # 5. IF vs group delay: large TB vs small TB
    # ------------------------------------------------------------------ #
    section("5. IF vs group delay — large TB (almost identical) vs small TB")
    t = np.arange(2, 256)

    sig1 = amgauss(256, 128, 90) * _sig(fmlin(256, 0, 0.5))
    tm1, T1 = loctime(sig1)
    fm1, B1 = locfreq(sig1)
    ifr1, t1 = inst_freq(sig1, t)
    f1 = np.linspace(0, 0.5 - 1 / 256, 256)
    gd1 = group_delay(sig1, f1)

    sig2 = amgauss(256, 128, 30) * _sig(fmlin(256, 0.2, 0.4))
    tm2, T2 = loctime(sig2)
    fm2, B2 = locfreq(sig2)
    ifr2, t2 = inst_freq(sig2, t)
    f2 = np.linspace(0.2, 0.4, 256)
    gd2 = group_delay(sig2, f2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7))
    ax1.plot(t1, ifr1, "*", label="instantaneous frequency")
    ax1.plot(gd1, f1, "-", label="group delay")
    ax1.set_xlim(0, 256)
    ax1.set_ylim(0, 0.5)
    ax1.set_title(f"Large TB: T*B = {T1 * B1:.2f} — curves almost superimposed")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Normalized frequency")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(t2, ifr2, "*", label="instantaneous frequency")
    ax2.plot(gd2, f2, "-", label="group delay")
    ax2.set_xlim(0, 256)
    ax2.set_ylim(0.2, 0.4)
    ax2.set_title(f"Small TB: T*B = {T2 * B2:.2f} — curves clearly different")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Normalized frequency")
    ax2.grid(True)
    ax2.legend()
    save_or_show(fig, "if_gd")
    print(f"\n  Large-TB signal: T*B = {T1 * B1:.3f}  (IF ~ GD)")
    print(f"  Small-TB signal: T*B = {T2 * B2:.3f}  (IF != GD)")
    pause()

    # ------------------------------------------------------------------ #
    # 6. Synthesis of mono-component non-stationary signals
    # ------------------------------------------------------------------ #
    section("6. Synthesis: amplitude modulation x frequency modulation")
    # 6a. Gaussian amplitude x linear FM
    sig1 = amgauss(256) * _sig(fmlin(256, 0, 0.5))
    show_time_spectrum(sig1, "Gaussian amplitude x linear FM")
    pause()
    # 6b. One-sided exponential (starts at t=100) x constant FM
    am2 = amexpos(256, t0=99, spread=32.0, kind="unilateral")
    sig2 = am2 * _sig(fmconst(256, 0.2))
    show_time_spectrum(sig2, "One-sided exponential x constant FM (0.2)")
    pause()
    # 6c. Doppler (moving target, 50 m/s, 10 m from observer, 4000 rpm, 200 Hz)
    fm3, am3, iflaw = doppler(256, 200, 4000 / 60, 10, 50)
    sig3 = am3 * fm3
    show_time_spectrum(sig3, "Doppler (car, 50 m/s, 10 m, 4000 rpm, 200 Hz)")
    pause()
    # 6d. Add complex colored Gaussian noise at -10 dB
    noise = noisecg(256, 0.8)
    sign = sigmerge(sig1, noise, -10)
    show_time_spectrum(sign, "Gaussian x FM + colored noise (-10 dB)")
    pause()

    # ------------------------------------------------------------------ #
    # 7. Multi-component signals
    # ------------------------------------------------------------------ #
    section("7. Multi-component: two chirps — IF/GD meaningless, STFT works")
    N = 128
    x = _sig(fmlin(N, 0, 0.2)) + _sig(fmlin(N, 0.3, 0.5))

    ifr, t_if = inst_freq(x)
    fnorm = np.arange(0, 0.5 + 1e-9, 0.01)
    gd = group_delay(x, fnorm)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))
    ax1.plot(t_if, ifr)
    ax1.set_xlim(0, N)
    ax1.set_ylim(0, 0.5)
    ax1.set_title("Instantaneous frequency (meaningless for 2 components)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Normalized frequency")
    ax1.grid(True)
    ax2.plot(gd, fnorm)
    ax2.set_xlim(0, N)
    ax2.set_ylim(0, 0.5)
    ax2.set_title("Group delay (meaningless for 2 components)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Normalized frequency")
    ax2.grid(True)
    save_or_show(fig, "multi_if_gd")
    pause()

    # STFT shows both components clearly
    tfr, t, f = ShortTimeFourierTransform(x).run()
    tfr, f = shift_tfr(tfr, f)
    show_tfr(tfr, t, f, "STFT of the two-chirp signal")
    pause()

    print("\nEnd of tfdemo2.")


if __name__ == "__main__":
    main()
