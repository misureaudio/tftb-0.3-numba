# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""tfdemo1 — Introduction to the Time-Frequency Toolbox.

Port of ``tfdemo1.m`` (O. Lemoine, May 1996, CNRS).

Demonstrates:

1. A linear FM signal: time domain, spectrum, Wigner-Ville distribution.
2. The same signal plus 0 dB complex Gaussian noise: the WVD still shows
   the linear frequency progression.
3. A bat sonar echo (``bat.mat``, Fe = 230.4 kHz): pseudo-WVD of the most
   interesting part — a narrow-band hyperbolic chirp from ~55 kHz to ~38 kHz.
4. A transient (one-sided exponential x constant FM) in -5 dB noise: the
   spectrogram localizes it at f = 0.25, t in [125, 160].

Usage::

    python demos/tfdemo1.py [--save <dir>] [--no-pause]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from tftb.generators import amexpos, fmconst, fmlin, noisecg, sigmerge
from tftb.processing.cohen import (PseudoWignerVilleDistribution,
                                   Spectrogram, WignerVilleDistribution)
from tftb.processing.utils import tftb_window

from demos._common import (init, load_mat, pause, save_or_show, section, sig,
                           shift_tfr, show_signal, show_spectrum, show_tfr,
                           show_time_spectrum)


def main():
    init()

    # ------------------------------------------------------------------ #
    # 1. Linear frequency modulation
    # ------------------------------------------------------------------ #
    section("1. Linear frequency modulation (fmlin: 0 -> 0.5)")
    sig1 = sig(fmlin(128, 0, 0.5))
    show_signal(sig1, "Linear frequency modulation")
    show_spectrum(sig1, "Spectrum")
    # From the time domain or the spectrum alone we cannot tell how the
    # frequency evolves in time. The Wigner-Ville distribution shows it:
    tfr, t, f = WignerVilleDistribution(sig1).run()
    # tfrwv axis is [0, 0.5) — no shift needed (WVD of a real signal is even).
    show_tfr(tfr, t, f, "Wigner-Ville distribution of the linear FM")
    pause()

    # ------------------------------------------------------------------ #
    # 2. Linear FM + 0 dB complex Gaussian noise
    # ------------------------------------------------------------------ #
    section("2. Linear FM + 0 dB complex Gaussian noise (sigmerge)")
    sig2 = sigmerge(sig1, noisecg(128), 0)
    show_time_spectrum(sig2, "Noisy linear FM")
    tfr2, t2, f2 = WignerVilleDistribution(sig2).run()
    show_tfr(tfr2, t2, f2, "WVD of the noisy signal (chirp still visible)")
    pause()

    # ------------------------------------------------------------------ #
    # 3. Bat sonar echo
    # ------------------------------------------------------------------ #
    section("3. Bat sonar echo (bat.mat, Fe = 230.4 kHz)")
    bat = load_mat("bat.mat")
    Fe = 230.4
    t_ms = np.arange(2048) / Fe
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t_ms, bat)
    ax.set_xlabel("Time [ms]")
    ax.set_title("Bat echo")
    ax.grid(True)
    save_or_show(fig, "bat")
    # Spectrum in dB (MATLAB: f0 = (-1024:1023)*Fe/2048 kHz, axis [0 Fe/2])
    dsp = np.fft.fftshift(np.abs(np.fft.fft(bat)) ** 2)
    f_khz = (np.arange(2048) - 1024) * Fe / 2048
    dsp = dsp / dsp.max()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(f_khz, 10 * np.log10(dsp + 1e-30))
    ax.set_xlim(0, Fe / 2)
    ax.set_ylim(-30, 0)
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("Squared modulus (dB)")
    ax.set_title("Spectrum (dB)")
    ax.grid(True)
    save_or_show(fig, "bat_spec")
    # Pseudo-WVD on the most interesting part (MATLAB:
    # tfrpwv(hilbert(bat), 8:8:2048, 256, tftb_window(127,'hanning'), 1))
    tfr_bat, t_bat, f_bat = PseudoWignerVilleDistribution(
        hilbert(bat),
        timestamps=np.arange(7, 2048, 8),
        n_fbins=256,
        fwindow=tftb_window(127, "hanning")).run()
    tfr_bat, f_bat = shift_tfr(tfr_bat, f_bat)
    show_tfr(tfr_bat, t_bat, f_bat, "Pseudo-WVD of the bat echo",
             xlabel="Time", ylabel="Normalized frequency")
    pause()

    # ------------------------------------------------------------------ #
    # 4. Noisy transient
    # ------------------------------------------------------------------ #
    section("4. Noisy transient (one-sided exponential x constant FM, -5 dB)")
    # MATLAB amexpo1s(92, 1, 64) == amexpos(92, t0=0, spread=64,
    # kind='unilateral'): amexpo1s uses a 1-based axis, tmt0 = (1:N) - t0,
    # and y = exp(-sqrt(pi)*tmt0/T) * (tmt0 >= 0).
    trans = amexpos(92, t0=0, spread=64.0, kind="unilateral") \
        * sig(fmconst(92))
    base = np.concatenate([np.zeros(100), trans, np.zeros(64)])
    sign = sigmerge(base, noisecg(256), -5)
    show_time_spectrum(sign, "Noisy transient")
    # The spectrogram localizes the transient at f = 0.25, t in [125, 160]:
    tfr_s, t_s, f_s = Spectrogram(sign).run()
    tfr_s, f_s = shift_tfr(tfr_s, f_s)
    show_tfr(tfr_s, t_s, f_s, "Spectrogram of the noisy transient")
    pause()

    print("\nEnd of tfdemo1.")


if __name__ == "__main__":
    main()
