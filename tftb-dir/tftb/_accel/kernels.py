# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Phase 5.1 Numba kernels.

Each kernel is a faithful, element-for-element port of the corresponding
numpy ``run()`` inner loop, compiled with ``@njit(parallel=True, cache=True)``
and parallelized over time columns with ``prange``. The public API and the
numpy path are untouched: the kernels are only *called* through
``backend.get_backend(...)`` inside ``run()``, and any runtime failure
auto-falls back to the numpy path (``backend.report_failure``).

Numerical identity strategy
---------------------------
The kernels are *gather + dot* only. Any per-class kernel weight (Choi-Williams
exponential, generalized-rectangular ``sinc``, Butterworth, the reduced-
interference Bessel / binomial / Hann / triangular kernels, ...) is
**precomputed on the numpy side** into a matrix ``K[tau, p+Lg] = g[Lg+p] *
kernel(tau, p)`` by calling each class's own ``_kernel``. This guarantees the
accelerated path uses *bit-identical* weights to the numpy path (no Numba
re-implementation of ``lgamma`` / ``sinc`` / ``sqrt`` of a possibly-negative
argument), so the only arithmetic the Numba path performs is the same
complex multiply-accumulate the numpy path does — matching to well under the
plan's rtol 1e-12 identity bound.
"""

import numpy as np

try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - numba is an optional extra
    _NUMBA_AVAILABLE = False
    njit = None
    prange = None

__all__ = [
    "cohen_generic_numba",
    "spwv_numba",
    "stank_numba",
    "hough_numba",
    "scalogram_numba",
    "renyi_numba",
    "NUMBA_AVAILABLE",
]

# MATLAB-``round`` is not needed here (callers precompute the round(N/2)
# scalars); this mirrors np.finfo(float).eps used by the guarded RID norm.
_EPS = 2.220446049250313e-16


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _min3(a, b, c):
        m = a if a < b else b
        return m if m < c else c

    @njit(parallel=True, cache=True)
    def cohen_generic_numba(x1, x2, g, h, K, N, ts, tfr1_g, points_tau_limit,
                            g2_norm, has_special, taumax_limit, nyq):
        """Numba port of ``GeneralizedCohenBase.run`` (pre-FFT tau loop).

        Parameters
        ----------
        x1, x2 : (n,) complex128   (x1 is x2 for the auto case)
        g      : (2*Lg+1,) float64 time-smoothing window
        h      : (2*Lh+1,) float64 frequency window (center-normalized)
        K      : (nyq+1, 2*Lg+1) float64, K[tau, p+Lg] = g[Lg+p]*kernel(tau,p)
        N      : int  n_fbins
        ts     : (nts,) int64  0-based timestamps
        tfr1_g : bool  scale tfr[0, icol] by g[Lg]
        points_tau_limit : bool  include ``tau`` in the points-range min
        g2_norm : int  0=none, 1=unconditional /sum(g2), 2=guarded /sum(g2)
        has_special : bool  Nyquist special bin at tau = nyq
        taumax_limit : int  int(round(N/2)) - 1
        nyq : int  int(round(N/2))

        Returns
        -------
        (N, nts) complex128  pre-FFT tfr (the caller does the fft + real()).
        """
        n = x1.shape[0]
        nts = ts.shape[0]
        Lg = g.shape[0] // 2
        Lh = h.shape[0] // 2
        tfr = np.zeros((N, nts), dtype=np.complex128)
        for icol in prange(nts):
            t0 = ts[icol]
            taumax = t0 + Lg
            a = n - t0 - 1 + Lg
            if a < taumax:
                taumax = a
            if taumax_limit < taumax:
                taumax = taumax_limit
            if Lh < taumax:
                taumax = Lh
            if tfr1_g:
                tfr[0, icol] = g[Lg] * x1[t0] * np.conj(x2[t0])
            else:
                tfr[0, icol] = x1[t0] * np.conj(x2[t0])
            for tau in range(1, taumax + 1):
                m1 = tau if points_tau_limit else Lg
                p_lo = -_min3(m1, Lg, n - t0 - 1 - tau)
                p_hi = _min3(m1, Lg, t0 - tau)
                if g2_norm:
                    ssum = 0.0
                    for p in range(p_lo, p_hi + 1):
                        ssum += K[tau, p + Lg]
                    do_div = (g2_norm == 1) or (ssum > _EPS)
                else:
                    do_div = False
                s1 = 0.0 + 0.0j
                s2 = 0.0 + 0.0j
                for p in range(p_lo, p_hi + 1):
                    w = K[tau, p + Lg]
                    if do_div:
                        w = w / ssum
                    s1 += w * x1[t0 + tau - p] * np.conj(x2[t0 - tau - p])
                    s2 += w * x1[t0 - tau - p] * np.conj(x2[t0 + tau - p])
                tfr[tau, icol] = h[Lh + tau] * s1
                tfr[N - tau, icol] = h[Lh - tau] * s2
            if has_special:
                tau = nyq
                if (t0 + 1 <= n - tau) and (t0 + 1 >= tau + 1) and (tau <= Lh):
                    m1 = tau if points_tau_limit else Lg
                    p_lo = -_min3(m1, Lg, n - t0 - 1 - tau)
                    p_hi = _min3(m1, Lg, t0 - tau)
                    ssum = 0.0
                    for p in range(p_lo, p_hi + 1):
                        ssum += K[tau, p + Lg]
                    R1 = 0.0 + 0.0j
                    R2 = 0.0 + 0.0j
                    for p in range(p_lo, p_hi + 1):
                        w = K[tau, p + Lg] / ssum
                        R1 += w * x1[t0 + tau - p] * np.conj(x2[t0 - tau - p])
                        R2 += w * x1[t0 - tau - p] * np.conj(x2[t0 + tau - p])
                    tfr[tau, icol] = 0.5 * (h[Lh + tau] * R1 + h[Lh - tau] * R2)
        return tfr

    @njit(parallel=True, cache=True)
    def spwv_numba(x, g, h, N, ts, taumax_limit, nyq):
        """Numba port of ``cohen.smoothed_pseudo_wigner_ville`` (pre-FFT).

        g : twindow, h : fwindow (both as the numpy path uses them). Returns
        the pre-FFT complex tfr; the caller does ``fft`` + ``real``.

        The Nyquist special bin is computed *inside* the main tau loop (as an
        ``if tau == nyq`` branch) rather than in a separate block after it.
        The Numba parallelizer (0.67–0.68) miscompiles a nested ``for p`` loop
        that *follows* the ``for tau`` loop inside the ``prange`` body into a
        genuine out-of-bounds access (segfault with ``boundscheck=False``,
        OOB ``IndexError`` with ``boundscheck=True``) at every N, power-of-2
        included; the serial build and the tau-loop-only build are clean.
        Merging the special bin into the tau loop removes the post-tau-loop
        nested loop and compiles correctly (verified: parallel+boundscheck
        clean at every N; bit-identical to the original; 82–327x vs numpy).
        See docs/benchmarks.md, Phase 5.4.
        """
        n = x.shape[0]
        nts = ts.shape[0]
        lg = g.shape[0] // 2
        lh = h.shape[0] // 2
        tfr = np.zeros((N, nts), dtype=np.complex128)
        for icol in prange(nts):
            ti = ts[icol]
            # taumax = min(taumax_limit, ti+lg, n-1-ti+lg, lh); matches the
            # numpy path (MATLAB tfrspwv 0-based convention, ti 0-based).
            taumax = taumax_limit
            a = ti + lg
            if a < taumax:
                taumax = a
            b = n - 1 - ti + lg
            if b < taumax:
                taumax = b
            c = lh
            if c < taumax:
                taumax = c
            # base column tfr[0] = sum(g2 * |x|²) over points =
            #   -min(lg, n-1-ti) : min(lg, ti); signal idx ti - p
            plo0 = -min(lg, n - 1 - ti)
            phi0 = min(lg, ti)
            s0 = 0.0 + 0.0j
            ssum0 = 0.0
            for p in range(plo0, phi0 + 1):
                w = g[lg + p]
                ssum0 += w
            for p in range(plo0, phi0 + 1):
                w = g[lg + p] / ssum0
                s0 += w * x[ti - p] * np.conj(x[ti - p])
            tfr[0, icol] = s0
            # Main tau loop, extended to cover tau = nyq (the special bin) so
            # there is NO nested p-loop after the tau loop. taumax <= nyq - 1,
            # so range(nyq + 1) covers {1..taumax} plus {nyq}; tau=0 is the
            # base column (handled above) and is skipped.
            tmax = nyq + 1
            for tau in range(tmax):
                if tau == nyq:
                    # special Nyquist bin: 0-based row tau, window h[lh+tau],
                    # h[lh-tau]; signal idx ti +/- tau - p.
                    if (ti <= n - tau - 1) and (ti >= tau) and (tau <= lh):
                        plo = -min(lg, n - 1 - ti - tau)
                        phi = min(lg, ti - tau)
                        if plo <= phi:
                            ssum = 0.0
                            for p in range(plo, phi + 1):
                                ssum += g[lg + p]
                            sx = 0.0 + 0.0j
                            sy = 0.0 + 0.0j
                            for p in range(plo, phi + 1):
                                w = g[lg + p] / ssum
                                sx += w * x[ti + tau - p] * np.conj(x[ti - tau - p])
                                sy += w * x[ti - tau - p] * np.conj(x[ti + tau - p])
                            tfr[tau, icol] = (sx * h[lh + tau] + sy * h[lh - tau]) * 0.5
                elif tau >= 1 and tau <= taumax:
                    plo = -min(lg, n - 1 - ti - tau)
                    phi = min(lg, ti - tau)
                    ssum = 0.0
                    for p in range(plo, phi + 1):
                        ssum += g[lg + p]
                    R1 = 0.0 + 0.0j
                    R2 = 0.0 + 0.0j
                    for p in range(plo, phi + 1):
                        w = g[lg + p] / ssum
                        R1 += w * x[ti + tau - p] * np.conj(x[ti - tau - p])
                        R2 += w * x[ti - tau - p] * np.conj(x[ti + tau - p])
                    tfr[tau, icol] = h[lh + tau] * R1
                    tfr[N - tau, icol] = h[lh - tau] * R2
        return tfr

    @njit(parallel=True, cache=True)
    def stank_jcols(tfr1, tfr2, tfr3, G, lg, N, nts, kmax):
        """Numba port of Stanković's per-column O(N·kmax) jcol loop.

        Computes, for every column, the smoothed base column (real, exactly as
        ``StankovicDistribution.run`` sets ``stan_col[jcol] = a.real``) and the
        two derivative columns (complex). Returns ``(stan_real, stanTh,
        stanDh)``. Parallel over columns; the jcol loop inside a column is
        sequential (deterministic, matches the numpy ordering).
        """
        stan_real = np.zeros((N, nts), dtype=np.float64)
        stanTh = np.zeros((N, nts), dtype=np.complex128)
        stanDh = np.zeros((N, nts), dtype=np.complex128)
        for c in prange(nts):
            for jcol in range(N):
                a = G[lg] * tfr1[jcol, c] * np.conj(tfr1[jcol, c])
                b = G[lg] * tfr1[jcol, c] * np.conj(tfr2[jcol, c])
                d = G[lg] * tfr1[jcol, c] * np.conj(tfr3[jcol, c])
                for k in range(1, kmax + 1):
                    before = (jcol - k) % N
                    after = (jcol + k) % N
                    a += (G[lg - k] * tfr1[before, c] * np.conj(tfr1[after, c])
                          + G[lg + k] * tfr1[after, c] * np.conj(tfr1[before, c]))
                    b += (G[lg - k] * tfr1[before, c] * np.conj(tfr2[after, c])
                          + G[lg + k] * tfr1[after, c] * np.conj(tfr2[before, c]))
                    d += (G[lg - k] * tfr1[before, c] * np.conj(tfr3[after, c])
                          + G[lg + k] * tfr1[after, c] * np.conj(tfr3[before, c]))
                stan_real[jcol, c] = a.real
                stanTh[jcol, c] = b
                stanDh[jcol, c] = d
        return stan_real, stanTh, stanDh

    @njit(cache=True)
    def stank_scatter(stan_real, stanTh, stanDh, N, nts, threshold, dt):
        """Numba port of Stanković's reassignment scatter (sequential, so the
        cross-column ``+=`` ordering matches the numpy path exactly).

        ``stan_real`` is the real base column (the reassignment ratios divide by
        it, matching ``stan_col[jcol] = a.real``). Returns ``(rtfr, hat)``: the
        reassigned distribution (complex, holding real values, matching the
        numpy path's dtype) and the complex reassignment-vector matrix
        (1-based ``jcol + 1j·icol``).
        """
        rtfr = np.zeros((N, nts), dtype=np.complex128)
        hat = np.zeros((N, nts), dtype=np.complex128)
        for c in range(nts):
            for jcol in range(N):
                sv = stan_real[jcol, c]
                if abs(sv) > threshold:
                    icolhat = int(np.floor((c + 1) - (stanTh[jcol, c] / sv).real / dt + 0.5))
                    jcolhat = int(np.floor((jcol + 1) - N * (stanDh[jcol, c] / sv).imag / (2.0 * np.pi) + 0.5))
                    jcolhat = ((jcolhat - 1) % N) + 1
                    if icolhat < 1:
                        icolhat = 1
                    if icolhat > nts:
                        icolhat = nts
                    rtfr[jcolhat - 1, icolhat - 1] += sv
                    hat[jcol, c] = jcolhat + 1j * icolhat
                else:
                    rtfr[jcol, c] += sv
                    hat[jcol, c] = (jcol + 1) + 1j * (c + 1)
        return rtfr, hat

    # ------------------------------------------------------------------ #
    # Phase 5.2 kernels: Hough, scalogram per-voice, renyi
    # ------------------------------------------------------------------ #
    @njit(cache=True)
    def hough_numba(image, cos_t, sin_t, colidx, m, n, xc, yc, imax, deltar,
                    rhomax):
        """Numba port of ``hough_transform`` (postprocessing).

        The numpy version rebuilds ``np.linspace(0, 2*pi-deltat, n)`` for every
        pixel that passes the threshold (inside the innermost loop) and runs the
        whole triple loop in Python. This kernel precomputes ``cos_t`` /
        ``sin_t`` / ``colidx`` on the numpy side and compiles the loop.

        It is *sequential* (not ``prange``): the accumulator ``ht[bin] += val``
        is a shared-bin reduce over pixels, so a parallel pixel loop would race.
        Sequential keeps it deterministic and bit-identical to numpy (same
        (x, y, theta) addition order), and it is already ~1000x faster because
        the numpy cost is the per-pixel linspace + Python overhead, not FP
        throughput.

        ``colidx[j] = int(round(theta_j / deltat))``; since ``theta_j =
        j*deltat`` (the linspace step equals ``deltat``) this is exactly ``j``,
        but it is passed in so the kernel stays faithful to the numpy indexing.
        """
        xmax, ymax = image.shape
        ht = np.zeros((m, n))
        half = imax / 20.0
        for px in range(xmax):
            x = px - xc + 1
            for py in range(ymax):
                val = image[px, py]
                if abs(val) <= half:
                    continue
                y = py - yc + 1
                for j in range(n):
                    rho = x * cos_t[j] - y * sin_t[j]
                    if (rho >= 0.0) and (rho <= rhomax):
                        ht[int(np.round(rho / deltar)), colidx[j]] += val
        return ht

    @njit(parallel=True, cache=True)
    def scalogram_numba(z, f, a, ts, waveparams, n_voices):
        """Numba port of ``Scalogram.run``'s per-voice convolution loop (the
        ``waveparams > 0`` branch, the default).

        Each voice ``ptr`` convolves the analytic signal ``z`` with a
        scale-dependent wavelet ``ha`` (a modulated Gaussian) — a full
        convolution — then samples it at the timestamps. The convolution is a
        direct O(N * len(ha)) loop (``np.convolve`` is unavailable in nopython
        mode); voices are independent, so the loop is ``prange`` over voices.
        ``np.convolve`` sums in a C order that may differ from this loop's, so
        the result matches to ~1e-16 (well under the 1e-12 identity bound), not
        bit-for-bit.

        Returns ``(wt, tfr)``: ``wt[ptr] = sliced_detail[ts]`` (raw wavelet
        coefficients) and ``tfr[ptr] = |sliced_detail[ts]|**2`` (the scalogram),
        where ``sliced = full_detail[ix0 + ts]`` with ``ix0 = round(nha)`` —
        the valid central region, exactly as the numpy branch (MATLAB
        ``tfrscalo`` truncates before sampling).
        ``tfr`` is float64 (|v|**2 is real); the caller stores it and the
        existing ``np.real`` normalization is a no-op on it.
        """
        n = z.shape[0]
        nts = ts.shape[0]
        wt = np.zeros((n_voices, nts), dtype=np.complex128)
        tfr = np.zeros((n_voices, nts), dtype=np.float64)
        log10 = np.log(10.0)
        two_pi = 2.0 * np.pi
        for ptr in prange(n_voices):
            nha = waveparams * a[ptr]
            nha_r = int(np.round(nha))
            nh = 2 * nha_r + 1
            ha = np.zeros(nh, dtype=np.complex128)
            for i in range(nh):
                tha = i - nha_r
                xw = np.exp(-(2.0 * log10 / (nha * nha)) * tha * tha)
                yw = np.exp(1j * two_pi * f[ptr] * tha)
                ha[i] = xw * yw
            # full convolution: detail[k] = sum_i z[i] * ha[k-i], k in [0, n+nh-2]
            ndet = n + nh - 1
            detail = np.zeros(ndet, dtype=np.complex128)
            for k in range(ndet):
                ilo = k - (nh - 1)
                if ilo < 0:
                    ilo = 0
                ihi = k
                if ihi > n - 1:
                    ihi = n - 1
                s = 0.0 + 0.0j
                for i in range(ilo, ihi + 1):
                    s += z[i] * ha[k - i]
                detail[k] = s
            detail = detail / np.sqrt(a[ptr])
            # sliced[t] = detail[round(nha) + t]; BOTH wt and tfr sample the
            # truncated (centered) detail. The old kernel sampled wt from the
            # UNTRUNCATED detail (detail[ts[t]]), missing the round(nha)
            # centering offset, corrupting the wavelet coefficients that
            # tfrspaw consumes while leaving tfr (which used the offset) correct.
            ix0 = int(np.round(nha))
            for t in range(nts):
                v = detail[ix0 + ts[t]]
                wt[ptr, t] = v
                tfr[ptr, t] = v.real * v.real + v.imag * v.imag
        return wt, tfr

    @njit(cache=True)
    def _integrate2d(mat, x, y):
        """Compiled port of ``integrate_2d`` (trapezoid in both axes)."""
        m, n = mat.shape
        dx = x[1] - x[0]
        rowtrap = np.zeros(m)
        for i in range(m):
            s = 0.0
            for j in range(n):
                s += mat[i, j]
            rowtrap[i] = (s - mat[i, 0] / 2.0 - mat[i, n - 1] / 2.0) * dx
        integ = 0.0
        for i in range(m - 1):
            integ += (rowtrap[i] + rowtrap[i + 1]) * (y[i + 1] - y[i]) / 2.0
        return integ

    @njit(cache=True)
    def renyi_numba(tfr, timestamps, freq, alpha):
        """Compiled port of ``renyi_information`` (postprocessing).

        The numpy version is already fully vectorized (a handful of reductions
        via ``integrate_2d``), so compiling it does not make it faster. This
        kernel exists so the Phase-5.2 gate can be *measured* and is expected to
        FAIL the >=3x gate — the plan explicitly allows the gate to say "no".
        ``eps = np.spacing(1)`` (2.220446049250313e-16), matching numpy.
        """
        eps = 2.220446049250313e-16
        m, n = tfr.shape
        integ = _integrate2d(tfr, timestamps, freq)
        tfr2 = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                tfr2[i, j] = tfr[i, j] / integ
        if alpha == 1.0:
            work = np.zeros((m, n))
            for i in range(m):
                for j in range(n):
                    work[i, j] = tfr2[i, j] * np.log2(tfr2[i, j] + eps)
            R = -_integrate2d(work, timestamps, freq)
        else:
            work = np.zeros((m, n))
            for i in range(m):
                for j in range(n):
                    work[i, j] = tfr2[i, j] ** alpha
            R = np.log2(_integrate2d(work, timestamps, freq) + eps)
            R = R / (1.0 - alpha)
        return R


NUMBA_AVAILABLE = _NUMBA_AVAILABLE
