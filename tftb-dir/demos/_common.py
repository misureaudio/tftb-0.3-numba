# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Shared utilities for the tftb demo suite.

Provides:

- ``init()`` — CLI parsing (``--save <dir>`` batch mode, ``--no-pause``)
- ``load_mat(name, var)`` — load a variable from a MATLAB ``.mat`` file
- ``section(title)`` / ``pause()`` — interactive flow control
- ``show_signal`` / ``show_spectrum`` / ``show_tfr`` / ``show_tfr_contour``
  / ``save_or_show`` — plotting helpers (save PNG in batch mode)
- ``run_frames`` — animate a list of TFR frames (save PNGs in batch mode)
"""

import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------
batch_mode = False
SAVE_DIR = None
NO_PAUSE = False
_fig_counter = [0]

#: Directory holding the MATLAB data files (bat.mat, gabor.mat). Override with
#: the TFTB_DATA_DIR environment variable.
DATA_DIR = os.environ.get("TFTB_DATA_DIR", r"D:\TOOLBOX\tftb-0.3\data")


def init(argv=None):
    """Parse demo CLI arguments. Call at the top of every demo script.

    Usage::

        python tfdemo1.py [--save <dir>] [--no-pause]
    """
    global batch_mode, SAVE_DIR, NO_PAUSE
    argv = argv if argv is not None else sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--save":
            batch_mode = True
            SAVE_DIR = argv[i + 1]
            os.makedirs(SAVE_DIR, exist_ok=True)
            matplotlib.use("Agg")
            i += 2
        elif argv[i] == "--no-pause":
            NO_PAUSE = True
            i += 1
        else:
            i += 1


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_mat(name, var=None):
    """Load a variable from a ``.mat`` file in :data:`DATA_DIR`.

    If ``var`` is None the first non-duplicate variable is returned.
    """
    path = os.path.join(DATA_DIR, name)
    d = loadmat(path, squeeze_me=True)
    if var is not None:
        return d[var]
    for k in d:
        if not k.startswith("__"):
            return d[k]
    raise KeyError(f"no data in {path}")


# ---------------------------------------------------------------------------
# Flow control
# ---------------------------------------------------------------------------
def sig(gen_result):
    """Extract the signal from a generator result.

    The port's FM generators (``fmlin``, ``fmconst``, ``fmhyp``, ``fmsin``)
    return ``(signal, iflaw)`` tuples; amplitude generators return plain
    arrays. Special cases at the call site:

    - ``doppler`` returns ``(fm, am, iflaw)`` — the signal is ``fm * am``;
    - ``gdpower`` returns ``(signal, gd, ...)`` — the signal is element 0.
    """
    if isinstance(gen_result, tuple) and len(gen_result) == 2:
        return gen_result[0]
    return np.asarray(gen_result)


def section(title):
    """Print a section header and pause (interactive mode only)."""
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")
    if not batch_mode and not NO_PAUSE:
        input("\nPress Enter to continue...")


def pause(msg="Press Enter to continue..."):
    if not batch_mode and not NO_PAUSE:
        input(msg)


def note(msg):
    """Print an informational note (e.g. a skipped section)."""
    print(f"\n[NOTE] {msg}")


# ---------------------------------------------------------------------------
# Figure bookkeeping
# ---------------------------------------------------------------------------
def _next_name(prefix="fig"):
    _fig_counter[0] += 1
    return f"{prefix}_{_fig_counter[0]:02d}"


def save_or_show(fig, prefix="fig", close=True):
    """Save the figure to PNG (batch mode) or display it (interactive).

    Returns the output path (batch) or figure name (interactive).
    """
    name = _next_name(prefix)
    if batch_mode:
        path = os.path.join(SAVE_DIR, f"{name}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        if close:
            plt.close(fig)
        return path
    plt.show()
    if close:
        plt.close(fig)
    return name


def save_current(prefix="fig"):
    """Save the *current* (``gcf``) figure in batch mode; no-op interactively.

    For third-party tftb plotting functions (e.g. ``plotifl``) that create
    their own figure and call ``plt.show()`` internally — the window is
    already shown interactively, so we only capture it in batch mode.
    """
    if not batch_mode:
        return None
    return save_or_show(plt.gcf(), prefix)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _tfr_display(tfr):
    """Return a real array suitable for display (|tfr| for complex TFRs)."""
    return np.abs(tfr) if np.iscomplexobj(tfr) else np.asarray(tfr, float)


def shift_tfr(tfr, f):
    """Shift the frequency axis so ``f`` runs from -1/2 to +1/2.

    The port returns TFR rows in the ``fft`` order (f=0 first); MATLAB's
    ``tfr*`` outputs are ``fftshift``ed. This reproduces the MATLAB
    orientation for plotting.
    """
    return np.fft.fftshift(tfr, axes=0), np.fft.fftshift(np.asarray(f, float))


def _pcolormesh(ax, t, f, d, cmap, vmin=None, vmax=None):
    """pcolormesh with correct shading for same-sized C and mesh axes."""
    kw = dict(cmap=cmap, shading="nearest")
    if vmin is not None:
        kw["vmin"] = vmin
    if vmax is not None:
        kw["vmax"] = vmax
    return ax.pcolormesh(t, f, d, **kw)


def show_tfr(tfr, t, f, title="", xlabel="Time", ylabel="Normalized frequency",
             cmap="inferno", prefix="tfr", log=False, vmin=None, vmax=None):
    """Plot a TFR matrix (n_fbins x n_time) with pcolormesh."""
    fig, ax = plt.subplots(figsize=(7, 5))
    d = _tfr_display(tfr)
    if log:
        d = np.log10(np.maximum(d, np.max(d) * 1e-10))
    pc = _pcolormesh(ax, t, f, d, cmap, vmin, vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("auto")
    fig.colorbar(pc, ax=ax, label="log10" if log else "value")
    return save_or_show(fig, prefix)


def show_tfr_contour(tfr, t, f, title="", prefix="tfr", nlevels=5,
                     xlabel="Time", ylabel="Frequency"):
    """Plot a TFR matrix as filled contours (MATLAB ``contour(t,f,tfr,V)``)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    d = _tfr_display(tfr)
    vmax = np.max(d)
    levels = np.linspace(0.1 * vmax, vmax, nlevels)
    cs = ax.contourf(t, f, d, levels=levels, cmap="inferno")
    fig.colorbar(cs, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("auto")
    return save_or_show(fig, prefix)


def show_signal(x, title="", xlabel="Time", ylabel="Amplitude", prefix="sig"):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(np.real(x))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    return save_or_show(fig, prefix)


def show_spectrum(x, title="", xlabel="Normalized frequency", prefix="spec",
                  db=False, f=None):
    """Plot the energy spectrum |fft(x)|^2 (optionally in dB)."""
    N = len(x)
    dsp = np.fft.fftshift(np.abs(np.fft.fft(x)) ** 2)
    if f is None:
        # Monotonic axis matching the fftshifted spectrum: -1/2 .. 1/2-1/N.
        f = (np.arange(N) - N // 2) / N
    fig, ax = plt.subplots(figsize=(7, 3))
    if db:
        ax.plot(f, 10 * np.log10(dsp / dsp.max() + 1e-30))
        ax.set_ylabel("Squared modulus (dB)")
    else:
        ax.plot(f, dsp)
        ax.set_ylabel("Squared modulus")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True)
    return save_or_show(fig, prefix)


def show_time_spectrum(x, title, xlabel="Time", db=False, prefix="ts"):
    """Two-panel figure: signal + its spectrum (MATLAB subplot(211/212) pattern)."""
    N = len(x)
    dsp = np.fft.fftshift(np.abs(np.fft.fft(x)) ** 2)
    f = (np.arange(N) - N // 2) / N
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))
    ax1.plot(np.real(x))
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Real part")
    ax1.set_title(title + " — signal")
    ax1.grid(True)
    if db:
        ax2.plot(f, 10 * np.log10(dsp / dsp.max() + 1e-30))
        ax2.set_ylabel("Squared modulus (dB)")
    else:
        ax2.plot(f, dsp)
        ax2.set_ylabel("Squared modulus")
    ax2.set_xlabel("Normalized frequency")
    ax2.set_title(title + " — spectrum")
    ax2.grid(True)
    return save_or_show(fig, prefix)


def show_curves(curves, title="", xlabel="Time", ylabel="Normalized frequency",
                prefix="curves"):
    """Plot one or more (x, y) curves with labels.

    ``curves``: list of (x, y, label) tuples.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for x, y, lab in curves:
        ax.plot(x, y, label=lab)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    if len(curves) > 1:
        ax.legend()
    return save_or_show(fig, prefix)


# ---------------------------------------------------------------------------
# Animation (replaces MATLAB movie() + the mov*.m helpers)
# ---------------------------------------------------------------------------
def run_frames(tfr_list, t, f, title="", prefix="anim"):
    """Animate a list of TFR frames (each n_fbins x n_time).

    ``f`` may be a single 1-D frequency axis shared by every frame, or a list
    of per-frame axes (movies that mix distributions with different
    frequency conventions — e.g. spectrogram frames followed by WVD frames).

    Batch mode: save every frame as a PNG. Interactive mode: matplotlib
    FuncAnimation.
    """
    # Normalize f to a list of per-frame axes.
    if isinstance(f, (list, tuple)) and len(f) == len(tfr_list) \
            and np.ndim(f[0]) == 1:
        f_list = [np.asarray(x, float) for x in f]
    else:
        f_list = [np.asarray(f, float)] * len(tfr_list)

    vmax = max(float(np.max(_tfr_display(x))) for x in tfr_list)
    if batch_mode:
        for i, tfr in enumerate(tfr_list):
            fig, ax = plt.subplots(figsize=(7, 5))
            _pcolormesh(ax, t, f_list[i], _tfr_display(tfr), "inferno", 0, vmax)
            ax.set_title(f"{title} (frame {i + 1}/{len(tfr_list)})")
            ax.set_xlabel("Time")
            ax.set_ylabel("Frequency")
            ax.set_aspect("auto")
            name = f"{prefix}_frame{i + 1:02d}"
            path = os.path.join(SAVE_DIR, f"{name}.png")
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    im = _pcolormesh(ax, t, f_list[0], _tfr_display(tfr_list[0]), "inferno", 0, vmax)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    ax.set_aspect("auto")
    fig.colorbar(im, ax=ax)

    from matplotlib.animation import FuncAnimation

    def update(i):
        im.set_array(_tfr_display(tfr_list[i]).ravel())
        return [im]

    anim = FuncAnimation(fig, update, frames=len(tfr_list), interval=200,
                         blit=False)
    plt.show()
    plt.close(fig)
    del anim
