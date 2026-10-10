# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""tfdemo3 — Linear time-frequency representations.

Port of ``tfdemo3.m`` (O. Lemoine, May 1996, CNRS).

Demonstrates:

1. The STFT of the word "GABOR" (``gabor.mat``, 338 samples @ 1 kHz):
   time domain, spectrum, and the squared-modulus STFT (the syllables
   "GA" and "BOR" are visible in the time-frequency plane).
2. The STFT time/frequency resolution tradeoff: a Dirac analysis window
   (perfect time, null frequency) vs a constant window (perfect frequency,
   null time).
3. The influence of the analysis-window shape/length: a long Hamming window
   (65) gives good frequency resolution but the two transient atoms are
   indistinguishable in time; a short Hamming window (17) separates them.
4. The Gabor representation of a linear chirp: the "bristling"
   biorthonormal window at Q=1 vs the smooth one at Q=4, and the much more
   readable Gabor coefficients.
5. The spectrogram: interference terms between two close parallel chirps
   (short vs long window), and the absence of interference for distant
   chirps.
6. The scalogram (Morlet wavelet): a Dirac pulse (cone in the
   time-frequency plane) and two sinusoids (frequency resolution depends on
   frequency).

Usage::

    python demos/tfdemo3.py [--save <dir>] [--no-pause]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib.pyplot as plt
import numpy as np
from tftb.generators import amgauss, anapulse, atoms, fmconst, fmlin
from tftb.processing.affine import Scalogram
from tftb.processing.cohen import Spectrogram
from tftb.processing.linear import ShortTimeFourierTransform, gabor
from tftb.processing.utils import tftb_window

from demos._common import (init, load_mat, pause, save_or_show, section,
                           shift_tfr, show_spectrum, show_tfr, show_time_spectrum)
from demos._common import sig as _sig


def _plot_gabor(tfr, n_half, n_time, title, prefix):
    """Plot a Gabor coefficient matrix (freq, time) — positive-freq half."""
    t = np.arange(1, n_time + 1)
    f = np.linspace(0, 0.5, n_half)
    fig, ax = plt.subplots(figsize=(7, 5))
    # tfr is (freq, time) = (n_freq, n_time); positive half is the first
    # n_half rows. pcolormesh expects C = (len(f), len(t)).
    pc = ax.pcolormesh(t, f, tfr[:n_half, :], cmap="inferno", shading="nearest")
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized frequency")
    ax.set_title(title)
    ax.set_aspect("auto")
    fig.colorbar(pc, ax=ax, label="value")
    return save_or_show(fig, prefix)


def main():
    init()

    # ------------------------------------------------------------------ #
    # 1. STFT of the word GABOR
    # ------------------------------------------------------------------ #
    section("1. Short-Time Fourier Transform of the word GABOR (1 kHz)")
    gabor_sig = load_mat("gabor.mat", "gabor")
    time = np.arange(338)
    show_time_spectrum(gabor_sig, "Word GABOR", xlabel="Time [ms]")
    # Spectrum in Hz (MATLAB: freq = (-169:168)/338*1000)
    dsp = np.fft.fftshift(np.abs(np.fft.fft(gabor_sig)) ** 2)
    freq = (np.arange(338) - 169) / 338 * 1000
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(freq, dsp)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Squared modulus")
    ax.set_title("Energy spectrum")
    ax.grid(True)
    save_or_show(fig, "gabor_spec")
    # STFT (MATLAB loaded a precomputed tfr; the port computes it in ~10 ms)
    tfr, t, f = Spectrogram(
        gabor_sig, timestamps=np.arange(338), n_fbins=256,
        fwindow=tftb_window(61, "hanning")).run()
    # Show only the positive-frequency half (0..0.5 -> 0..500 Hz)
    tfr_p = tfr[:128, :]
    f_p = (np.arange(128) / 256) * 1000
    fig, ax = plt.subplots(figsize=(7, 5))
    pc = ax.pcolormesh(time, f_p, tfr_p, cmap="inferno", shading="nearest")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Squared modulus of the STFT of the word GABOR")
    ax.set_aspect("auto")
    fig.colorbar(pc, ax=ax, label="value")
    save_or_show(fig, "gabor_stft")
    print("\n  The 'GA' syllable (~30-60 ms, ~150 Hz) and the 'BOR' syllable")
    print("  (~150-250 ms, mean frequency decreasing 140->110 Hz) are visible.")
    pause()

    # ------------------------------------------------------------------ #
    # 2. STFT time/frequency resolution tradeoff
    # ------------------------------------------------------------------ #
    section("2. STFT resolution tradeoff: Dirac window vs constant window")
    sig = amgauss(128) * _sig(fmlin(128))
    # 2a. Dirac window (h=1): perfect time resolution, null frequency resolution
    tfr, t, f = ShortTimeFourierTransform(
        sig, timestamps=np.arange(128), n_fbins=128,
        fwindow=np.array([1.0])).run()
    tfr, f = shift_tfr(tfr, f)
    show_tfr(tfr, t, f, "STFT with a Dirac window (h=1): perfect time, null frequency")
    pause()
    # 2b. Constant window (h=ones(127)): perfect frequency, null time
    tfr, t, f = ShortTimeFourierTransform(
        sig, timestamps=np.arange(128), n_fbins=128,
        fwindow=np.ones(127)).run()
    tfr, f = shift_tfr(tfr, f)
    show_tfr(tfr, t, f, "STFT with a constant window (h=ones(127)): perfect frequency, null time")
    pause()

    # ------------------------------------------------------------------ #
    # 3. Influence of the analysis-window shape/length
    # ------------------------------------------------------------------ #
    section("3. Two transient atoms — long vs short Hamming window")
    sig = atoms(128, np.array([[45, .25, 32, 1], [85, .25, 32, 1]]))
    # Long window: good frequency resolution, poor time resolution
    h1 = tftb_window(65, "hamming")
    tfr, t, f = ShortTimeFourierTransform(
        sig, timestamps=np.arange(128), n_fbins=128, fwindow=h1).run()
    tfr, f = shift_tfr(tfr, f)
    show_tfr(tfr, t, f, "Hamming window (65): good frequency, two atoms merged in time")
    pause()
    # Short window: poor frequency resolution, good time resolution
    h2 = tftb_window(17, "hamming")
    tfr, t, f = ShortTimeFourierTransform(
        sig, timestamps=np.arange(128), n_fbins=128, fwindow=h2).run()
    tfr, f = shift_tfr(tfr, f)
    show_tfr(tfr, t, f, "Hamming window (17): poorer frequency, two atoms separated in time")
    pause()

    # ------------------------------------------------------------------ #
    # 4. Gabor representation
    # ------------------------------------------------------------------ #
    section("4. Gabor representation of a linear chirp (Q=1 vs Q=4)")
    N1 = 256
    Ng = 33
    sig = _sig(fmlin(N1))
    g = tftb_window(Ng, "gauss")
    g = g / np.linalg.norm(g)
    # Q=1: critical sampling — bristling biorthonormal window
    Q = 1
    tfr, dgr, h = gabor(sig, 16, Q, g)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(np.arange(1, N1 + 1), h)
    ax.set_xlim(1, N1)
    ax.set_ylim(-0.3, 0.55)
    ax.set_title(f"Biorthonormal window (Q={Q}) — bristling")
    ax.set_xlabel("Time")
    ax.grid(True)
    save_or_show(fig, "gabor_h_q1")
    _plot_gabor(tfr, 8, 16, f"Squared modulus of the Gabor coefficients (Q={Q})",
                "gabor_q1")
    pause()
    # Q=4: oversampling — smooth biorthonormal window, readable coefficients
    Q = 4
    tfr, dgr, h = gabor(sig, 32, Q, g)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(np.arange(1, N1 + 1), h)
    ax.set_xlim(1, N1)
    ax.set_ylim(-0.01, 0.09)
    ax.set_title(f"Biorthonormal window (Q={Q}) — smooth")
    ax.set_xlabel("Time")
    ax.grid(True)
    save_or_show(fig, "gabor_h_q4")
    _plot_gabor(tfr, 16, 32, f"Squared modulus of the Gabor coefficients (Q={Q})",
                "gabor_q4")
    pause()

    # ------------------------------------------------------------------ #
    # 5. Spectrogram: interference terms
    # ------------------------------------------------------------------ #
    section("5. Spectrogram — interference between close vs distant chirps")
    # 5a. Two close parallel chirps
    sig = _sig(fmlin(128, 0, 0.4)) + _sig(fmlin(128, 0.1, 0.5))
    h1 = tftb_window(23, "gauss")
    tfr, t, f = Spectrogram(sig, n_fbins=128, fwindow=h1).run()
    tfr, f = shift_tfr(tfr, f)
    show_tfr(tfr, t, f, "Two close chirps — short window (h=23 gauss)")
    h2 = tftb_window(63, "gauss")
    tfr, t, f = Spectrogram(sig, n_fbins=128, fwindow=h2).run()
    tfr, f = shift_tfr(tfr, f)
    show_tfr(tfr, t, f, "Two close chirps — long window (h=63 gauss)")
    pause()
    # 5b. Two distant parallel chirps
    sig = _sig(fmlin(128, 0, 0.3)) + _sig(fmlin(128, 0.2, 0.5))
    tfr, t, f = Spectrogram(sig, n_fbins=128, fwindow=h1).run()
    tfr, f = shift_tfr(tfr, f)
    show_tfr(tfr, t, f, "Two distant chirps — short window (h=23 gauss)")
    tfr, t, f = Spectrogram(sig, n_fbins=128, fwindow=h2).run()
    tfr, f = shift_tfr(tfr, f)
    show_tfr(tfr, t, f, "Two distant chirps — long window (h=63 gauss)")
    pause()

    # ------------------------------------------------------------------ #
    # 6. Scalogram (Morlet wavelet)
    # ------------------------------------------------------------------ #
    section("6. Scalogram — Dirac pulse and two sinusoids")
    # 6a. Dirac pulse at t0=64: cone in the time-frequency plane
    sig1 = anapulse(128)
    tfr, ts, f, wt = Scalogram(sig1, fmin=0.05, fmax=0.45, n_voices=64,
                               waveparams=6).run()
    fig, ax = plt.subplots(figsize=(7, 5))
    pc = ax.pcolormesh(ts, f, tfr, cmap="inferno", shading="nearest")
    ax.set_yscale("log")
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency (log)")
    ax.set_title("Scalogram of a Dirac pulse (cone in time-frequency)")
    ax.set_aspect("auto")
    fig.colorbar(pc, ax=ax, label="value")
    save_or_show(fig, "scalo_pulse")
    pause()
    # 6b. Two sinusoids: frequency resolution increases with frequency
    sig2 = _sig(fmconst(128, .15)) + _sig(fmconst(128, .35))
    tfr, ts, f, wt = Scalogram(sig2, fmin=0.05, fmax=0.45, n_voices=128,
                               waveparams=6).run()
    fig, ax = plt.subplots(figsize=(7, 5))
    pc = ax.pcolormesh(ts, f, tfr, cmap="inferno", shading="nearest")
    ax.set_yscale("log")
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency (log)")
    ax.set_title("Scalogram of two sinusoids (resolution depends on frequency)")
    ax.set_aspect("auto")
    fig.colorbar(pc, ax=ax, label="value")
    save_or_show(fig, "scalo_2sin")
    pause()

    print("\nEnd of tfdemo3.")


if __name__ == "__main__":
    main()
