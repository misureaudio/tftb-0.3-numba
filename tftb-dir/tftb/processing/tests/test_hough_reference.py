# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Transcribed reference tests for the Hough transform (htl).

Direct transcription of ``tftb-0.3/tests/htlt.m`` (O. Lemoine, 1995). A
binary image is built with ``lineplot`` (ported here, it is test-only in
MATLAB), passed through ``hough_transform`` at a finer rho/theta resolution,
and the peak location is checked against the planted line.

Cases:
  1. one slant line (rho=20, theta=pi/4), 64x64 image, 128x128 resolution.
  2. four axis-aligned lines (rho=10, theta in {0, pi/2, pi, 3pi/2}).
  3. one slant line (rho=20, theta=pi/4), 63x63 image, 127x127 resolution.
"""
import numpy as np
import pytest

from tftb.processing.postprocessing import hough_transform


def lineplot(rho, theta, M, N):
    """Python port of tftb-0.3/tests/lineplot.m — binary image of lines.

    ``rho``/``theta`` (same length) parametrize the lines; ``IM`` is an
    (N, M) binary image with the origin at the image center.
    """
    rho = np.atleast_1d(rho)
    theta = np.atleast_1d(theta)
    assert len(rho) == len(theta), "RHO and THETA must have the same length"
    IM = np.zeros((N, M))
    Xc = (N + 1) / 2 if N % 2 else N / 2
    X0, Xf = 1 - Xc, Xc - 1 if N % 2 else Xc
    Yc = (M + 1) / 2 if M % 2 else M / 2
    Y0, Yf = 1 - Yc, Yc - 1 if M % 2 else Yc
    for k in range(len(rho)):
        if abs(np.sin(theta[k])) < 1000 * np.finfo(float).eps:
            # near-vertical line (sin ~ 0): solve for x given y
            for y in np.arange(Y0, Yf + 1):
                x = int(round((rho[k] + y * np.sin(theta[k])) / np.cos(theta[k])))
                if X0 <= x <= Xf:
                    IM[int(x + Xc - 1), int(y + Yc - 1)] = 1
        else:
            # solve for y given x
            for x in np.arange(X0, Xf + 1):
                y = int(round((-rho[k] + x * np.cos(theta[k])) / np.sin(theta[k])))
                if Y0 <= y <= Yf:
                    IM[int(x + Xc - 1), int(y + Yc - 1)] = 1
    return IM


def test_htl_slant_line_even():
    M = N = 64
    rho, theta = 20, np.pi / 4
    IM = lineplot(rho, theta, M, N)
    Mr = Nt = 128
    ht, r, t = hough_transform(IM, m=Mr, n=Nt)
    tmax = int(np.argmax(np.max(ht, axis=0)))  # theta of the global peak
    rmax = int(np.argmax(np.max(ht, axis=1)))  # rho of the global peak
    assert abs(r[rmax] - rho) <= 10 / Mr
    assert abs(t[tmax] - theta) <= 10 / Nt
    Max = np.max(ht)
    assert np.sum(ht > Max / 5) <= 1  # a single peak


def test_htl_four_lines():
    M = N = 64
    rho = np.array([10, 10, 10, 10])
    theta = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    IM = lineplot(rho, theta, M, N)
    Mr = Nt = 128
    ht, r, t = hough_transform(IM, m=Mr, n=Nt)
    rmax = int(np.argmax(np.max(ht, axis=1)))
    assert abs(r[rmax] - rho[0]) <= 10 / Mr
    Max = np.max(ht)
    assert np.sum(ht > Max / 5) <= 4  # at most four peaks
    # all four planted lines reach the same peak height
    maxis = []
    for k in range(4):
        th = np.nonzero(np.abs(t - theta[k]) < 0.01)[0]
        maxis.append(np.max(ht[rmax, th]))
    assert np.allclose(maxis, Max, atol=1000 * np.finfo(float).eps)


def test_htl_slant_line_odd():
    M = N = 63
    rho, theta = 20, np.pi / 4
    IM = lineplot(rho, theta, M, N)
    Mr = Nt = 127
    ht, r, t = hough_transform(IM, m=Mr, n=Nt)
    tmax = int(np.argmax(np.max(ht, axis=0)))
    rmax = int(np.argmax(np.max(ht, axis=1)))
    assert abs(r[rmax] - rho) <= 100 / Mr
    assert abs(t[tmax] - theta) <= 10 / Nt
    Max = np.max(ht)
    assert np.sum(ht > Max / 5) <= 1  # a single peak
