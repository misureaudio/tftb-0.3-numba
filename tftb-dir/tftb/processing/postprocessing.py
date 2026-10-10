#! /usr/bin/env python
# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""
Postprocessing functions.
"""

import numpy as np
from tftb.processing.utils import integrate_2d
from tftb._accel import backend as _accel_backend


def hough_transform(image, m=None, n=None):
    """hough_transform

    :param image:
    :param m:
    :param n:
    :type image:
    :type m:
    :type n:
    :return:
    :rtype:
    """
    image = np.asarray(image)
    xmax, ymax = image.shape
    if m is None:
        m = xmax
    if n is None:
        n = ymax

    rhomax = np.sqrt((xmax ** 2) + (ymax ** 2)) / 2.0
    deltar = rhomax / (m - 1.0)
    deltat = 2 * np.pi / n
    imax = np.amax(image)

    if xmax % 2 != 0:
        xc = (xmax + 1) // 2
    else:
        xc = xmax // 2
    if ymax % 2 != 0:
        yc = (ymax + 1) // 2
    else:
        yc = ymax // 2

    # Phase 5.2: Numba port of the triple (x, y, theta) loop. The numpy version
    # rebuilds np.linspace(0, 2*pi-deltat, n) for EVERY pixel that passes the
    # threshold (inside the innermost loop); the kernel precomputes the theta
    # grid once and compiles the loop. It iterates all pixels in row-major order
    # (the same (x, y) order as the numpy loop, since x = px - xc + 1 is
    # monotone in px), so the shared-bin `+=` accumulation is bit-identical.
    if _accel_backend.get_backend("hough", xmax) == "numba":
        try:
            from tftb._accel import kernels as _k
            theta_grid = np.linspace(0, 2 * np.pi - deltat, n)
            cos_t = np.cos(theta_grid)
            sin_t = np.sin(theta_grid)
            colidx = np.rint(theta_grid / deltat).astype(np.int64)
            ht = _k.hough_numba(image, cos_t, sin_t, colidx, m, n,
                                xc, yc, imax, deltar, rhomax)
            rho = np.linspace(0, rhomax, m)
            theta = theta_grid
            return ht, rho, theta
        except Exception:
            _accel_backend.report_failure("hough")

    ht = np.zeros((m, n))
    if xmax % 2 != 0:
        xf = xc - 1
    else:
        xf = xmax // 2
    x0 = 1 - xc
    if ymax % 2 != 0:
        yf = yc - 1
    else:
        yf = ymax // 2
    y0 = 1 - yc
    for x in range(x0, xf + 1):
        for y in range(y0, yf + 1):
            if np.abs(image[x + xc - 1, y + yc - 1]) > imax / 20.0:
                for theta in np.linspace(0, 2 * np.pi - deltat, n):
                    rho = x * np.cos(theta) - y * np.sin(theta)
                    if (rho >= 0) and (rho <= rhomax):
                        ht[int(np.round(rho / deltar)),
                           int(np.round(theta / deltat))] += image[x + xc - 1,
                                                                   y + yc - 1]
    # MATLAB: rho = 0:deltar:rhomax (M points); theta = 0:deltat:2*pi-deltat
    # (N points). The radial axis has M samples, not N.
    rho = np.linspace(0, rhomax, m)
    theta = np.linspace(0, 2 * np.pi - deltat, n)
    return ht, rho, theta


def imextrac(image):
    """Faithful port of tftb-0.3 ``imextrac.m`` — 4-connected dot labeling.

    Labels the connected components (4-connectivity: up/down/left/right) of a
    binary image. Background is 0; dot labels are assigned in MATLAB's order
    (first row left-to-right, then first column top-to-bottom, then raster),
    starting at 2 (MATLAB ``NbDots`` starts at 1, the first dot gets 2). When
    two already-labeled neighbors meet, the higher label is renamed to the
    lower (union-by-rename), so the surviving labels depend on this exact
    order — which is why this is a faithful port rather than a
    ``scipy.ndimage.label`` call.

    :param image: 2-D array treated as binary (nonzero = foreground).
    :return: (labels, nb_dots) — integer label array (same shape) and the
        highest dot label (0 if no foreground).
    """
    image2 = (np.asarray(image) != 0).astype(int)
    nbrow, nbcol = image2.shape
    nb_dots = 1  # 1-based; the first dot is labeled 2
    if image2[0, 0] == 1:
        nb_dots += 1
        image2[0, 0] = nb_dots
    for i in range(1, nbcol):
        if image2[0, i] == 1:
            if image2[0, i - 1] > 1:
                image2[0, i] = image2[0, i - 1]
            else:
                nb_dots += 1
                image2[0, i] = nb_dots
    for j in range(nbrow):
        if image2[j, 0] == 1:
            if j > 0 and image2[j - 1, 0] > 1:
                image2[j, 0] = image2[j - 1, 0]
            else:
                nb_dots += 1
                image2[j, 0] = nb_dots
        for i in range(1, nbcol):
            if image2[j, i] == 1:
                above = image2[j - 1, i] if j > 0 else 0
                left = image2[j, i - 1]
                if above == 0 and left == 0:
                    nb_dots += 1
                    image2[j, i] = nb_dots
                elif above == 0 and left > 1:
                    image2[j, i] = left
                elif above > 1 and left == 0:
                    image2[j, i] = above
                elif above > 1 and left > 1:
                    mindot = min(above, left)
                    maxdot = max(above, left)
                    image2[image2 == maxdot] = mindot
                    image2[j, i] = mindot
                # else: impossible — processed neighbors are 0 or > 1.
    return image2, nb_dots


def energy_dots(tfr, threshold=5, keep=10):
    """Faithful port of tftb-0.3 ``tfrsurf.m`` — extract the biggest energy dots.

    Thresholds the TFR at ``threshold`` percent of its maximum, labels the
    resulting binary image's dots (``imextrac``), ranks the dots by pixel
    count (descending), and keeps the ``keep`` biggest.

    :param tfr: time-frequency representation (2-D).
    :param threshold: energy threshold in percent of the maximum (default 5).
    :param keep: number of biggest dots to keep (default 10).
    :return: (tfr2, ordered_surfaces) — ``tfr2`` is the TFR with only the
        ``keep`` biggest dots (the rest zeroed), and ``ordered_surfaces`` is
        the pixel count of every dot, largest first.
    """
    the_max = np.max(tfr)
    binary = (tfr >= threshold * the_max * 0.01)
    labels, nb_dots = imextrac(binary)
    # pixel count per dot (index 0 = background), matching MATLAB's loop.
    surfaces = np.bincount(labels.ravel(), minlength=nb_dots + 1)
    # MATLAB: [OS,Idx]=sort(Surfaces(2:end)) ascending (stable), then fliplr
    # -> descending; on equal sizes the larger dot index comes first.
    sizes = surfaces[1:]
    order = np.argsort(sizes, kind="stable")[::-1]
    ordered_surfaces = sizes[order]
    dot_labels = order + 1  # position p in sizes is dot p+1.
    binary_keep = np.zeros(tfr.shape, dtype=int)
    for i in range(min(keep, nb_dots)):
        binary_keep[labels == dot_labels[i]] = 1
    tfr2 = tfr * binary_keep
    return tfr2, ordered_surfaces


def renyi_information(tfr, timestamps=None, freq=None, alpha=3.0):
    """renyi_information

    :param tfr:
    :param timestamps:
    :param freq:
    :param alpha:
    :type tfr:
    :type timestamps:
    :type freq:
    :type alpha:
    :return:
    :rtype:
    """
    if alpha == 1 and tfr.min().min() < 0:
        raise ValueError("Distribution with negative values not allowed.")
    m, n = tfr.shape
    if timestamps is None:
        timestamps = np.arange(n) + 1
    elif np.allclose(timestamps, np.arange(n)):
        timestamps += 1
    if freq is None:
        freq = np.arange(m)
    freq.sort()
    tfr = np.asarray(tfr, dtype=float)

    # Phase 5.2: compiled port of the reduction. The numpy version is already
    # fully vectorized (a handful of integrate_2d reductions), so this is NOT
    # expected to clear the >=3x gate — it exists so the gate can be measured
    # and is expected to FAIL (the plan explicitly allows the gate to say "no").
    if _accel_backend.get_backend("renyi", n) == "numba":
        try:
            from tftb._accel import kernels as _k
            return _k.renyi_numba(tfr, timestamps.astype(float),
                                  freq.astype(float), float(alpha))
        except Exception:
            _accel_backend.report_failure("renyi")

    tfr = tfr / integrate_2d(tfr, timestamps, freq)
    if alpha == 1:
        R = -integrate_2d(tfr * np.log2(tfr + np.spacing(1)), timestamps, freq)
    else:
        R = np.log2(integrate_2d(tfr ** alpha, timestamps, freq) + np.spacing(1))
        R = R / (1 - alpha)
    return R


def ideal_tfr(iflaws, timestamps=None, n_fbins=None):
    """ideal_tfr

    :param iflaws:
    :param timestamps:
    :param n_fbins:
    :type iflaws:
    :type timestamps:
    :type n_fbins:
    :return:
    :rtype:
    """
    ifrow = iflaws.shape[0]
    if timestamps is None:
        timestamps = np.arange(iflaws[0, :].shape[0])
    if n_fbins is None:
        n_fbins = iflaws[0, :].shape[0]

    tcol = timestamps.shape[0]

    tfr = np.zeros((n_fbins, tcol))
    for icol in range(tcol):
        ti = timestamps[icol]
        for fi in range(ifrow):
            if np.isnan(iflaws[fi, ti]):
                tfr[ti, fi] = np.nan
            else:
                tfr[int(np.round(iflaws[fi, ti] * 2 * (n_fbins - 1))), icol] = 1
    freqs = np.arange(n_fbins, dtype=float) / n_fbins * 0.5
    return tfr, timestamps, freqs


def friedman_density(tfr, re_mat, timestamps=None):
    """friedman_density

    :param tfr:
    :param re_mat:
    :param timestamps:
    :type tfr:
    :type re_mat:
    :type timestamps:
    :return:
    :rtype:
    """
    tfrrow, tfrcol = tfr.shape
    if timestamps is None:
        timestamps = np.arange(tfrcol)

    tifd = np.zeros((tfrrow, tfrcol))
    bins = 0.5 + np.arange(tfrrow)
    threshold = np.sum(np.sum(tfr)) * 0.5 / tfr.size

    for j in range(tfrcol):
        indices = tfr[:, j] > threshold
        if np.any(indices):
            # MATLAB hist(x, v) with bin CENTERS v=0.5+arange(N): an integer
            # value v lands in 1-based bin v (probe-verified: value 31 -> bin
            # 31). The old np.histogram built edges from bin midpoints, which
            # shifted every value one bin high. real(hat) for reassigned bins
            # is an integer 1-based freq bin; sub-threshold bins carry inf and
            # are ignored by MATLAB hist (out of bin range) — drop them here.
            vals = np.real(re_mat[indices, j])
            finite = np.isfinite(vals)
            if np.any(finite):
                bin0 = np.rint(vals[finite]).astype(int) - 1  # -> 0-based
                keep = (bin0 >= 0) & (bin0 < tfrrow)
                if np.any(keep):
                    occ = np.bincount(bin0[keep], minlength=tfrrow)
                    tifd[:tfrrow, j] = occ
    tifd = tifd / np.sum(np.sum(tifd))
    return tifd


def ridges(tfr, re_mat, timestamps=None, method='rsp'):
    """ridges

    :param tfr:
    :param re_mat:
    :param timestamps:
    :param method:
    :type tfr:
    :type re_mat:
    :type timestamps:
    :type method:
    :return:
    :rtype:
    """
    method = method.lower()
    tfrrow, tfrcol = tfr.shape
    if timestamps is None:
        timestamps = np.arange(tfrcol)

    n_fbins = tfrrow
    # MATLAB: frequencies = (1:Nf)'  (1-based row labels). The reassignment
    # vectors in `re_mat` (hat) are also 1-based, so the stationary test must
    # compare against 1-based labels (the old 0-based `freqs`/`icol` were a
    # systematic off-by-one).
    freqs = np.arange(1, n_fbins + 1)
    threshold = np.sum(np.sum(tfr)) * 0.5 / tfr.size

    if method in ('rpwv', 'rpmh'):
        # frequency-only: hat is a real 1-based freq bin. Stationary where the
        # reassigned bin equals the original row. pointsf = indices/(2*Nf).
        scale = 2.0 * n_fbins

        def stationary(col, icol1):
            return (tfr[:, col] > threshold) & (re_mat[:, col] == freqs)
    elif method == 'rspwv':
        # 2-axis: hat is complex (jcolhat + 1j*icolhat, 1-based). Stationary
        # where both the freq and time reassignments are identity. pointsf =
        # indices/(2*Nf).
        scale = 2.0 * n_fbins

        def stationary(col, icol1):
            return ((tfr[:, col] > threshold)
                    & (np.real(re_mat[:, col]) == freqs)
                    & (np.imag(re_mat[:, col]) == icol1))
    elif method in ('rsp', 'type1'):
        # 2-axis, same stationary test as rspwv but pointsf = indices/Nf.
        scale = n_fbins

        def stationary(col, icol1):
            return ((tfr[:, col] > threshold)
                    & (np.real(re_mat[:, col]) == freqs)
                    & (np.imag(re_mat[:, col]) == icol1))
    else:
        raise ValueError("Unknown time frequency representation.")

    # MATLAB accumulates across columns: pointst=[pointst;...], pointsf=[...;
    # ...]. The old code reassigned each iteration and kept only the last
    # column.
    time_parts = []
    freq_parts = []
    for icol in range(tfrcol):
        ti = timestamps[icol]
        mask = stationary(icol, icol + 1)
        if np.any(mask):
            indices = np.nonzero(mask)[0] + 1  # 1-based row indices
            time_parts.append(np.full(indices.shape, ti))
            freq_parts.append(indices / scale)
    if time_parts:
        time_points = np.concatenate(time_parts)
        freq_points = np.concatenate(freq_parts)
    else:
        time_points = np.array([])
        freq_points = np.array([])
    return time_points, freq_points


if __name__ == '__main__':
    from tftb.generators import atoms
    from tftb.processing import Spectrogram
    s = atoms(64, np.array([[32, 0.3, 16, 1]]))
    spec = Spectrogram(s)
    tfr, t, f = spec.run()
    print(renyi_information(tfr, t, f, 3))
