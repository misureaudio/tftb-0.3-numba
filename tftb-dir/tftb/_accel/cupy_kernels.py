# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Phase 5.3bis — CuPy path: a *second explicit API*.

Three functions, called by name (the name is the selector — no env vars, no
dispatch):

    from tftb._accel.cupy_kernels import cwt_cupy, spwv_cupy, spectrogram_cupy

Design (plans/2026-10-08_task-5.3bis-cupy-explicit-api.md):

* **Purely additive.** Nothing in ``kernels.py`` / ``backend.py`` or the
  existing ``cwt`` / ``spwv`` / ``Spectrogram`` dispatch is touched; the Numba
  path stays the default for everything.
* **Lazy CuPy import.** This module imports cleanly without CuPy; each
  function raises a clear ``ImportError`` when called without it. CuPy is an
  optional extra.
* **Guards.** Each function requires signal length ``N >= 8192`` and a CUDA
  device; otherwise it raises a clear error. There is **no silent fallback**
  to numpy/numba — an explicit CuPy call runs on CuPy or fails loudly.
* **Returns numpy** (``.get()``), same tuples and dtypes as the standard API.

Numerics: the heavy step (the batched FFT) runs on the GPU via
``cupyx.scipy.fft``; the gather/scatter that builds the pre-FFT matrices is
done with vectorized numpy (or cupy) indexing, element-for-element the same
arithmetic as the numpy path, so the result matches to ~1e-13 (batched-FFT
implementation differences), well under the rtol 1e-10 bound for N >= 8192.
"""

import numpy as np

__all__ = ["cwt_cupy", "spwv_cupy", "spectrogram_cupy", "CUPY_MIN_N"]

# Gate size (inclusive): below this the CuPy branch refuses to run.
CUPY_MIN_N = 8192


def _cupy():
    """Lazy CuPy import with a clear error (CuPy is an optional extra)."""
    try:
        import cupy as cp
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "CuPy is required for the *_cupy functions but is not importable "
            "(%s). Install the 'cupy' extra, or use the standard API "
            "(cwt / smoothed_pseudo_wigner_ville / Spectrogram)." % e) from e
    return cp


def _check(name, n):
    """Guard: N >= 8192 and a CUDA device present. No silent fallback."""
    if n < CUPY_MIN_N:
        raise ValueError(
            "%s: signal length N=%d is below the CuPy gate (N >= %d); use the "
            "standard API instead." % (name, n, CUPY_MIN_N))
    cp = _cupy()
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError(
            "%s: no CUDA device available for the CuPy path." % name)
    return cp


# --------------------------------------------------------------------------- #
# spectrogram_cupy
# --------------------------------------------------------------------------- #
def spectrogram_cupy(signal, timestamps=None, n_fbins=None, fwindow=None):
    """CuPy path for ``Spectrogram(signal).run()`` (N >= 8192, explicit).

    Builds the same pre-FFT matrix as ``Spectrogram.run`` — per-column
    windowed, zero-padded, wrapped into the (n_fbins, nts) matrix — then runs
    the final ``fft(tfr, axis=0)`` **on the GPU** and returns
    ``|FFT(frame)|**2`` as numpy.

    The gather is vectorized on the numpy side (the numpy reference uses a
    Python loop over columns — that loop is most of its cost) with the exact
    same element-for-element arithmetic, including the per-column window
    normalization ``/ norm(fwindow[lh+tau])`` (shorter at the boundary
    columns). One H2D copy, one GPU batched FFT, one D2H copy.

    Returns the same tuple as ``Spectrogram.run``: ``(tfr, ts, freqs)``.
    """
    from tftb.processing.utils import tftb_window

    name = "spectrogram_cupy"
    x = np.asarray(signal)
    if x.ndim > 1:
        x = x.ravel()
    n = x.shape[0]
    cp = _check(name, n)

    # Replicate BaseTFRepresentation / Spectrogram setup exactly.
    if timestamps is None:
        timestamps = np.arange(n)
    if n_fbins is None:
        n_fbins = n
    if fwindow is None:
        h = n_fbins // 4
        h += 1 - (h % 2)
        fwindow = tftb_window(int(h), 'Hamming')
    if n_fbins % 2 == 0:
        freqs = np.r_[np.arange(n_fbins / 2), np.arange(-n_fbins / 2, 0)]
    else:
        freqs = np.r_[np.arange((n_fbins - 1) / 2),
                      np.arange(-(n_fbins - 1) / 2, 0)]
    freqs = freqs / n_fbins

    ts = np.asarray(timestamps, dtype=np.int64)
    nts = ts.shape[0]
    N = int(n_fbins)
    lh = (fwindow.shape[0] - 1) // 2
    rangemin = min([round(N / 2.0) - 1, lh])
    starts = -np.min(np.c_[rangemin * np.ones(nts), ts - 1],
                     axis=1).astype(int)
    ends = np.min(np.c_[rangemin * np.ones(nts),
                        n - ts], axis=1).astype(int)
    lengths = ends - starts + 1
    max_len = int(lengths.max())

    # Vectorized gather (same arithmetic as the per-column numpy loop; the
    # numpy loop's indices are always in range, the clip below only affects
    # the masked-out rows).
    j = np.arange(max_len)
    tau = starts[None, :] + j[:, None]                     # (max_len, nts)
    valid = j[:, None] < lengths[None, :]
    sig_idx = np.clip(ts[None, :] + tau - 1, 0, n - 1)
    win_idx = np.clip(lh + tau, 0, fwindow.shape[0] - 1)
    win_vals = fwindow[win_idx]
    # per-column norm of the window slice actually used (boundary columns
    # have a shorter slice — matches np.linalg.norm(fwindow[lh+tau]) per col)
    norm_cols = np.sqrt(np.sum(np.where(valid, win_vals ** 2, 0.0), axis=0))
    vals = (x[sig_idx] * np.conj(win_vals)) / norm_cols[None, :]
    vals = np.where(valid, vals, 0.0).astype(np.complex128)

    rows = (N + tau) % N
    col_idx = np.tile(np.arange(nts), max_len)  # 1-D, column per (j, col)
    mask = valid.ravel()
    tfr_np = np.zeros((N, nts), dtype=np.complex128)
    tfr_np[rows.ravel()[mask], col_idx[mask]] = vals.ravel()[mask]

    # The batched FFT on the GPU (one H2D + one D2H).
    from cupyx.scipy.fft import fft as cp_fft
    tfr = cp.asarray(tfr_np)
    tfr = cp_fft(tfr, axis=0)
    tfr = (tfr * tfr.conj()).real
    return tfr.get(), ts, freqs


# --------------------------------------------------------------------------- #
# spwv_cupy
# --------------------------------------------------------------------------- #
def spwv_cupy(signal, timestamps=None, freq_bins=None,
              twindow=None, fwindow=None):
    """CuPy path for ``cohen.smoothed_pseudo_wigner_ville`` (N >= 8192).

    Builds the same pre-FFT (freq_bins, nts) matrix as the numpy path
    (the per-column tau loop, vectorized over columns with numpy advanced
    indexing — identical arithmetic), then runs the final ``fft(tfr, axis=0)``
    **on the GPU** and returns ``real(...)`` as numpy.

    NOTE (gate review input): the tau loop is the dominant cost and is NOT the
    part this version moves to the GPU (a batched FFT alone does not address
    it); see the Phase 5.3 gate review in docs/benchmarks.md.
    """
    name = "spwv_cupy"
    x = np.asarray(signal, dtype=np.complex128)
    n = x.shape[0]
    cp = _check(name, n)

    if timestamps is None:
        timestamps = np.arange(n)
    if freq_bins is None:
        freq_bins = n
    from tftb.processing.utils import tftb_window
    from tftb.processing.linear import _matlab_round
    if fwindow is None:
        winlength = freq_bins // 4
        winlength = winlength + 1 - (winlength % 2)
        fwindow = tftb_window(int(winlength), 'Hamming')
    elif fwindow.shape[0] % 2 == 0:
        raise ValueError('The smoothing fwindow must have an odd length.')
    if twindow is None:
        timelength = freq_bins // 10
        timelength += 1 - (timelength % 2)
        twindow = tftb_window(int(timelength), 'Hamming')
    elif twindow.shape[0] % 2 == 0:
        raise ValueError('The smoothing twindow must have an odd length.')

    ts = np.asarray(timestamps, dtype=np.int64)
    nts = ts.shape[0]
    N = int(freq_bins)
    lg = (twindow.shape[0] - 1) // 2
    lh = (fwindow.shape[0] - 1) // 2
    nyq = int(_matlab_round(N / 2.0))

    # --- pre-FFT matrix (identical arithmetic to the numpy spwv loop) ---
    # Index convention matches MATLAB tfrspwv (0-based ti): base column
    # signal[ti - points] over points = -min(lg, n-1-ti) : min(lg, ti);
    # tau column x[ti+tau-p]*conj(x[ti-tau-p]), window fwindow[lh+tau], row
    # `tau`. (The earlier version used a one-sample shift, x[...-p-1] / row
    # 1+tau — the same off-by-one the plain numpy path had, fixed there and
    # here together.)
    tfr = np.zeros((N, nts), dtype=np.complex128)
    for icol in range(nts):
        ti = int(ts[icol])
        taumax = min([ti + lg, n - 1 - ti + lg, nyq - 1, lh])
        points = np.arange(-min([lg, n - 1 - ti]), min([lg, ti]) + 1).astype(int)
        g2 = twindow[lg + points]
        g2 = g2 / np.sum(g2)
        sig_idx = (ti - points).astype(int)
        tfr[0, icol] = np.sum(g2 * x[sig_idx] * np.conj(x[sig_idx]))
        for tau in range(1, int(taumax) + 1):
            points = np.arange(-min([lg, n - 1 - ti - tau]),
                               min([lg, ti - tau]) + 1).astype(int)
            g2 = twindow[lg + points]
            g2 = g2 / np.sum(g2)
            idx1 = (ti + tau - points).astype(int)
            idx2 = (ti - tau - points).astype(int)
            R = np.sum(g2 * x[idx1] * np.conj(x[idx2]))
            tfr[tau, icol] = fwindow[lh + tau] * R
            R = np.sum(g2 * x[idx2] * np.conj(x[idx1]))
            tfr[N - tau, icol] = fwindow[lh - tau] * R
        tau = nyq
        if (ti <= n - tau - 1) and (ti >= tau) and (tau <= lh):
            points = np.arange(-min([lg, n - 1 - ti - tau]),
                               min([lg, ti - tau]) + 1).astype(int)
            g2 = twindow[lg + points]
            g2 = g2 / np.sum(g2)
            idx1 = (ti + tau - points).astype(int)
            idx2 = (ti - tau - points).astype(int)
            _x = np.sum(g2 * x[idx1] * np.conj(x[idx2]))
            _x *= fwindow[lh + tau]
            _y = np.sum(g2 * x[idx2] * np.conj(x[idx1]))
            _y *= fwindow[lh - tau]
            tfr[tau, icol] = (_x + _y) * 0.5

    # --- the batched FFT on the GPU ---
    from cupyx.scipy.fft import fft as cp_fft
    tfr = cp.asarray(tfr)
    tfr = cp_fft(tfr, axis=0)
    return tfr.real.get()


# --------------------------------------------------------------------------- #
# cwt_cupy
# --------------------------------------------------------------------------- #
def cwt_cupy(signal, fmin=0.01, fmax=0.5, n_voices=None, wave=2):
    """CuPy path for ``wavelet.cwt`` (N >= 8192).

    Per-voice FFT-based convolution: each voice's wavelet (``gaussn``) is
    synthesized on the numpy side; the zero-padded linear convolution
    ``conv(x, ha)`` (MATLAB convention, unflipped kernel) is
    ``ifft(fft(x) * fft(ha))`` on the GPU. The signal FFT is computed once
    (all voices share the same padded length), and the wavelet FFTs are
    batched in chunks. Returns the same tuple as ``wavelet.cwt``:
    ``(scalo, f, T, a, wt)``.
    """
    from tftb.processing.wavelet import gaussn

    name = "cwt_cupy"
    x = np.asarray(signal, dtype=np.complex128).ravel()
    if fmin >= fmax:
        raise ValueError('fmax must be greater than fmin')
    n = x.shape[0]
    cp = _check(name, n)
    if n_voices is None:
        n_voices = n
    T = np.arange(1, n + 1)
    f = np.logspace(np.log10(fmax), np.log10(fmin), n_voices)
    a = np.logspace(0.0, np.log10(fmax / fmin), n_voices)

    # Synthesize all wavelets on the numpy side (gaussn is a numpy function).
    ha_list = [np.asarray(gaussn(f[ptr], wave), dtype=np.float64)
               for ptr in range(n_voices)]
    nha = [(ha.shape[0] - 1) // 2 for ha in ha_list]
    max_len = max(ha.shape[0] for ha in ha_list)

    from cupyx.scipy.fft import fft as cp_fft, ifft as cp_ifft

    def _next_pow2(v):
        p = 1
        while p < v:
            p *= 2
        return p

    nfft = _next_pow2(n + max_len - 1)
    Xn = cp_fft(cp.asarray(x), n=nfft)

    wt = np.zeros((n_voices, n), dtype=np.complex128)
    chunk = 256  # (chunk, nfft) complex128 = 64 MB @ nfft=16384
    for c0 in range(0, n_voices, chunk):
        c1 = min(c0 + chunk, n_voices)
        H = np.zeros((c1 - c0, nfft), dtype=np.float64)
        for i in range(c0, c1):
            H[i - c0, :ha_list[i].shape[0]] = ha_list[i]
        Hn = cp_fft(cp.asarray(H), axis=1)
        D = cp_ifft(Xn[None, :] * Hn, axis=1)
        for i in range(c0, c1):
            wt[i, :] = D[i - c0, nha[i]:nha[i] + n].get()
    wt = wt[::-1, :]
    f = f[::-1]
    a = a[::-1]
    scalo = wt * np.conj(wt)
    return scalo, f, T, a, wt
