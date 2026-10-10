# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Zak transform and its inverse.

Faithful port of the ``tftb-0.3`` Zak-transform pair:

* ``zak``  <- ``zak.m``   (discrete Zak transform)
* ``izak`` <- ``izak.m``  (inverse discrete Zak transform)

The Zak transform reshapes a length-``N1`` signal into an ``M x N`` matrix
(column-major, as MATLAB) and applies a length-``N`` FFT to each row, scaled
by ``1/sqrt(N)``.  It is unitary (Parseval: it preserves the inner product and
the energy) and its inverse ``izak`` reconstructs the signal exactly when
``N1`` is a multiple of both ``M`` and ``N``.

MATLAB conventions matched:

* ``reshape(sig, M, N)`` is **column-major** -> ``np.reshape(sig, (M, N),
  order='F')``.
* ``dzt = fft(Msig.')/sqrt(N)``: ``Msig.'`` is the (non-conjugate) transpose,
  so the FFT runs along the ``N`` axis of each of the ``M`` rows ->
  ``np.fft.fft(Msig.T, axis=0) / sqrt(N)``.
* The default ``[N, M] = divider(N1)`` uses MATLAB's ``divider`` algorithm
  (``zak.m`` line 12), which differs from the standalone ``tftb.utils.divider``
  utility (e.g. ``divider(12)`` -> MATLAB ``[3,4]`` vs the utility ``(4,3)``).
  The MATLAB algorithm is reproduced here so the default factorization matches
  the oracle exactly.
"""
import numpy as np


def _divider(N1):
    """MATLAB ``divider.m``: two factors of ``N1`` closest to ``sqrt(N1)``.

    Reproduces the certified algorithm (``N=floor(sqrt(N1))`` then iterate
    ``M=ceil(N1/N); N=floor(N1/M)`` to a fixed point).  This is **not** the
    same as ``tftb.utils.divider`` for every input (they agree on 10, 256,
    315 but disagree on 12: MATLAB ``[3,4]``, utility ``(4,3)``), so it is
    kept local to match the ``zak`` oracle.
    """
    N = int(np.floor(np.sqrt(N1)))
    while True:
        Nold = N
        M = int(np.ceil(N1 / N))
        N = int(np.floor(N1 / M))
        if N == Nold:
            break
    return N, M


def zak(sig, N=None, M=None):
    """Discrete Zak transform of ``sig``.

    Port of ``zak.m``.  Reshapes the length-``N1`` signal into an ``M x N``
    matrix (column-major) and FFTs each row, scaled by ``1/sqrt(N)``.  The
    result ``dzt`` has shape ``(N, M)``.

    Parameters
    ----------
    sig : (N1,) array_like
        Signal (``N1 >= 3``).
    N, M : int, optional
        Zak dimensions with ``N * M == N1``.  If both are omitted, the default
        factorization ``[N, M] = divider(N1)`` (closest to ``sqrt(N1)``) is
        used.  If only one is given, the other is derived from ``N1``.

    Returns
    -------
    dzt : (N, M) complex
        The Zak transform.
    """
    sig = np.asarray(sig).ravel()
    N1 = sig.shape[0]
    if N1 <= 2:
        raise ValueError("SIG must have at least 3 elements")

    if N is None and M is None:
        N, M = _divider(N1)
    elif M is not None and N is None:
        N = N1 // M
    elif N is not None and M is None:
        M = N1 // N
    if (M <= 1) or (N <= 1) or (M >= N1) or (N >= N1):
        raise ValueError("M and N must be between 2 and N1-1")
    if (N1 % M != 0) or (N1 % N != 0):
        raise ValueError("M and N must be such that N1 is a multiple of M and N")

    Msig = np.reshape(sig, (M, N), order="F")      # column-major (MATLAB)
    dzt = np.fft.fft(Msig.T, axis=0) / np.sqrt(N)   # fft(Msig.')/sqrt(N)
    return dzt


def izak(dzt):
    """Inverse discrete Zak transform.

    Port of ``izak.m`` (moved here from ``tftb.utils``; a back-compatible
    ``utils.izak`` re-export is kept).  Reconstructs the length-``N*M`` signal
    from an ``(N, M)`` Zak transform: each column is IFFT'd (scaled by
    ``sqrt(N)``) and written to the corresponding strided positions.

    Parameters
    ----------
    dzt : (N, M) complex
        Zak transform from :func:`zak`.

    Returns
    -------
    sig : (N*M,) complex
        Reconstructed signal.
    """
    dzt = np.asarray(dzt)
    if dzt.ndim == 2:
        N, M = dzt.shape
    else:
        N, M = dzt.shape[0], 1
    sig = np.zeros(N * M, dtype=complex)
    for im in range(M):
        sig[im + np.arange(N) * M] = np.sqrt(N) * np.fft.ifft(dzt[:, im], axis=0)
    return sig
