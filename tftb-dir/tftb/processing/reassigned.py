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

"""Reassigned TF processing."""

import numpy as np
from tftb.processing.utils import derive_window, tftb_window


def _reassign_scatter(tfr, time_shift, freq_shift, threshold, n_fbins, tcol):
    """Shared reassignment scatter (the MATLAB ``tfrr*.m`` pattern).

    For each bin with ``|tfr| > threshold``, scatter its energy to the
    reassigned bin; sub-threshold bins stay in place. Two modes:

    * **2-axis** (``time_shift`` given, e.g. ``tfrrsp``/``tfrrgab``): energy
      moves to ``(jcol - freq_shift, icol + time_shift)`` — frequency wrapped
      modulo ``n_fbins``, time clamped to ``[0, tcol-1]``. ``hat`` is complex,
      ``jcolhat + 1j*icolhat`` in MATLAB's 1-based coordinates, with
      ``inf*(1+1j)`` at sub-threshold bins.
    * **frequency-only** (``time_shift is None``, e.g. ``tfrrpwv/pmh/ppag/
      spwv``): energy moves only in frequency, ``jcolhat = (jcol -
      freq_shift) % n_fbins``, time unchanged. ``hat`` is real, the 1-based
      frequency bin, with ``inf`` at sub-threshold bins.

    :param tfr: (n_fbins, tcol) real energy matrix (already ``|.|**2`` for the
        2-axis family, ``real(fft(...))`` — possibly negative — for the
        frequency-only family).
    :param time_shift: (n_fbins, tcol) real integer time reassignment, or
        ``None`` for frequency-only mode.
    :param freq_shift: (n_fbins, tcol) real integer frequency reassignment.
    :param threshold: energy threshold for reassignment.
    :return: (rtfr, hat)
    """
    freq_only = time_shift is None
    rtfr = np.zeros((n_fbins, tcol), dtype=complex)
    if freq_only:
        hat = np.zeros((n_fbins, tcol))  # real: 1-based freq bin / inf
    else:
        hat = np.zeros((n_fbins, tcol), dtype=complex)
    for icol in range(tcol):
        for jcol in range(n_fbins):
            if np.abs(tfr[jcol, icol]) > threshold:
                jcolhat = int(jcol - np.real(freq_shift[jcol, icol])) % n_fbins
                if freq_only:
                    rtfr[jcolhat, icol] += tfr[jcol, icol]
                    hat[jcol, icol] = jcolhat + 1
                else:
                    icolhat = int(icol + np.real(time_shift[jcol, icol]))
                    icolhat = min(max(icolhat, 0), tcol - 1)
                    rtfr[jcolhat, icolhat] += tfr[jcol, icol]
                    hat[jcol, icol] = (jcolhat + 1) + 1j * (icolhat + 1)
            else:
                hat[jcol, icol] = np.inf if freq_only else np.inf * (1 + 1j)
                rtfr[jcol, icol] += tfr[jcol, icol]
    return rtfr, hat


def pseudo_wigner_ville(signal, timestamps=None, n_fbins=None, fwindow=None):
    """pseudo_wigner_ville

    :param signal:
    :param timestamps:
    :param n_fbins:
    :param fwindow:
    :type signal:
    :type timestamps:
    :type n_fbins:
    :type fwindow:
:return:
:rtype:
    """
    xrow = signal.shape[0]
    if timestamps is None:
        timestamps = np.arange(signal.shape[0])
    if n_fbins is None:
        n_fbins = signal.shape[0]
    tcol = timestamps.shape[0]

    if fwindow is None:
        hlength = np.floor(n_fbins / 4.0)
        if hlength % 2 == 0:
            hlength += 1
        fwindow = tftb_window(int(hlength), 'Hamming')
    elif fwindow.shape[0] % 2 == 0:
        raise ValueError('The smoothing fwindow must have an odd length.')
    lh = (fwindow.shape[0] - 1) // 2
    fwindow = fwindow / fwindow[lh]

    tfr = np.zeros((n_fbins, tcol), dtype=complex)
    tf2 = np.zeros((n_fbins, tcol), dtype=complex)
    dh = derive_window(fwindow)
    for icol in range(tcol):
        ti = timestamps[icol]
        taumax = min([ti, xrow - 1 - ti, np.round(n_fbins / 2.0) - 1, lh])
        tau = np.arange(-taumax, taumax + 1)
        indices = np.remainder(n_fbins + tau, n_fbins) + 1
        tfr[indices - 1, icol] = fwindow[lh + tau] * signal[ti + tau] * \
            np.conj(signal[ti - tau])
        tf2[indices - 1, icol] = dh[lh + tau] * signal[ti + tau] * \
            np.conj(signal[ti - tau])
        tau = np.round(n_fbins / 2)
        if (ti <= (xrow - tau)) and (ti > (tau + 1)) and (tau <= lh):
            _x = fwindow[lh + 1 + tau] * signal[ti + tau] * np.conj(signal[ti - tau])
            _y = fwindow[lh + 1 - tau] * signal[ti - tau] * np.conj(signal[ti + tau])
            tfr[tau + 1, icol] = (_x + _y) * 0.5
            _x = dh[lh + 1 + tau] * signal[ti + tau] * np.conj(signal[ti - tau])
            _y = dh[lh + 1 - tau] * signal[ti - tau] * np.conj(signal[ti + tau])
            tf2[tau + 1, icol] = (_x + _y) * 0.5
    tfr = np.real(np.fft.fft(tfr, axis=0))
    tf2 = np.imag(np.fft.fft(tf2, axis=0))
    tfr = tfr.ravel()
    tf2 = tf2.ravel()
    no_warn_mask = tfr != 0
    tf2[no_warn_mask] *= n_fbins / tfr[no_warn_mask] / (2 * np.pi)
    tf2[no_warn_mask] = np.round(tf2[no_warn_mask])
    tfr = tfr.reshape(n_fbins, tcol)
    tf2 = tf2.reshape(n_fbins, tcol)

    tmin = timestamps.min()
    tmax = timestamps.max()
    rtfr, hat = _reassign_scatter(
        tfr, None, tf2, 1.0e-6 * (np.abs(signal[tmin:(tmax + 1)]) ** 2).mean(),
        n_fbins, tcol)
    return tfr, rtfr, hat


def pseudo_margenau_hill(signal, timestamps=None, n_fbins=None, fwindow=None):
    """pseudo_margenau_hill

    :param signal:
    :param timestamps:
    :param n_fbins:
    :param fwindow:
    :type signal:
    :type timestamps:
    :type n_fbins:
    :type fwindow:
:return:
:rtype:
    """
    xrow = signal.shape[0]
    if timestamps is None:
        timestamps = np.arange(signal.shape[0])
    if n_fbins is None:
        n_fbins = signal.shape[0]
    tcol = timestamps.shape[0]

    if fwindow is None:
        hlength = np.floor(n_fbins / 4.0)
        if hlength % 2 == 0:
            hlength += 1
        fwindow = tftb_window(int(hlength), 'Hamming')
    elif fwindow.shape[0] % 2 == 0:
        raise ValueError('The smoothing fwindow must have an odd length.')
    lh = (fwindow.shape[0] - 1) // 2
    fwindow = fwindow / fwindow[lh]

    tfr = np.zeros((n_fbins, tcol), dtype=complex)
    tf2 = np.zeros((n_fbins, tcol), dtype=complex)
    dh = derive_window(fwindow)
    tfr = np.zeros((n_fbins, tcol), dtype=complex)
    for icol in range(tcol):
        ti = timestamps[icol]
        start = min([np.round(n_fbins / 2.0) - 1, lh, xrow - 1 - ti])
        end = min([np.round(n_fbins / 2.0) - 1, lh, ti])
        tau = np.arange(-start, end + 1)
        indices = np.remainder(n_fbins + tau, n_fbins)
        tfr[indices, icol] = fwindow[lh + tau] * signal[ti] * \
            np.conj(signal[ti - tau])
        tf2[indices, icol] = dh[lh + tau] * signal[ti] * \
            np.conj(signal[ti - tau])

    tfr = np.fft.fft(tfr, axis=0)
    tf2 = np.fft.fft(tf2, axis=0)
    tfr = tfr.ravel()
    tf2 = tf2.ravel()
    no_warn_mask = tfr != 0
    tf2[no_warn_mask] *= n_fbins / tfr[no_warn_mask] / (2 * np.pi)
    tf2[no_warn_mask] = np.round(tf2[no_warn_mask])
    tfr = np.real(tfr)
    tf2 = np.imag(tf2)
    tfr = tfr.reshape(n_fbins, tcol)
    tf2 = tf2.reshape(n_fbins, tcol)

    rtfr, hat = _reassign_scatter(
        tfr, None, tf2, 1.0e-6 * (np.abs(signal) ** 2).mean(), n_fbins, tcol)
    return tfr, rtfr, hat


def pseudo_page(signal, timestamps=None, n_fbins=None, fwindow=None):
    """pseudo_page

    :param signal:
    :param timestamps:
    :param n_fbins:
    :param fwindow:
    :type signal:
    :type timestamps:
    :type n_fbins:
    :type fwindow:
:return:
:rtype:
    """
    if timestamps is None:
        timestamps = np.arange(signal.shape[0])
    if n_fbins is None:
        n_fbins = signal.shape[0]
    tcol = timestamps.shape[0]

    if fwindow is None:
        hlength = np.floor(n_fbins / 4.0)
        if hlength % 2 == 0:
            hlength += 1
        fwindow = tftb_window(int(hlength), 'Hamming')
    elif fwindow.shape[0] % 2 == 0:
        raise ValueError('The smoothing fwindow must have an odd length.')
    lh = (fwindow.shape[0] - 1) // 2
    fwindow = fwindow / fwindow[lh]

    tfr = np.zeros((n_fbins, tcol), dtype=complex)
    tf2 = np.zeros((n_fbins, tcol), dtype=complex)
    dh = derive_window(fwindow)

    for icol in range(tcol):
        tau = np.arange(min([n_fbins - 1, lh, icol]) + 1)
        indices = np.remainder(n_fbins + tau, n_fbins) + 1
        tfr[indices - 1, icol] = fwindow[lh + tau] * signal[icol] * \
            np.conj(signal[icol - tau])
        tf2[indices - 1, icol] = dh[lh + tau] * signal[icol] * \
            np.conj(signal[icol - tau])
        tf2[0, icol] += signal[icol] * np.conj(signal[icol])

    tfr = np.fft.fft(tfr, axis=0)
    tf2 = np.fft.fft(tf2, axis=0)
    tfr = tfr.ravel()
    tf2 = tf2.ravel()
    no_warn_mask = tfr != 0
    tf2[no_warn_mask] *= n_fbins / tfr[no_warn_mask] / (2 * np.pi)
    tf2[no_warn_mask] = np.round(tf2[no_warn_mask])
    tfr = np.real(tfr)
    tf2 = np.imag(tf2)
    tfr = tfr.reshape(n_fbins, tcol)
    tf2 = tf2.reshape(n_fbins, tcol)

    rtfr, hat = _reassign_scatter(
        tfr, None, tf2, 1.0e-6 * (np.abs(signal) ** 2).mean(), n_fbins, tcol)
    return tfr, rtfr, hat


def morlet_scalogram(signal, timestamps=None, n_fbins=None, tbp=0.25):
    """morlet_scalogram

    :param signal:
    :param timestamps:
    :param n_fbins:
    :param tbp:
    :type signal:
    :type timestamps:
    :type n_fbins:
    :type tbp:
:return:
:rtype:
    """
    xrow = signal.shape[0]
    if timestamps is None:
        timestamps = np.arange(signal.shape[0])
    if n_fbins is None:
        n_fbins = signal.shape[0]
    k = 0.001
    tcol = timestamps.shape[0]
    deltat = timestamps[1:] - timestamps[:-1]
    if deltat.min() != deltat.max():
        raise ValueError("Time instants must be regularly sampled.")
    else:
        dt = deltat.min()

    tfr = np.zeros((n_fbins, tcol), dtype=complex)
    tf2 = np.zeros((n_fbins, tcol), dtype=complex)
    M = np.ceil(tbp * n_fbins * np.sqrt(2 * np.log(1 / k)))
    tau = np.arange(M + int(np.round(n_fbins / 2)) + 1)
    hstar = np.exp(-(tau / (n_fbins * tbp)) ** 2 / 2.0) * \
        np.exp(-1j * 2 * np.pi * tau / n_fbins)
    thstar = tau * hstar

    # 0-based transcription of MATLAB tfrrmsc.m. `ti` is a 0-based sample
    # index (default timestamps = 0..N-1). MATLAB works in 1-based `ti` with
    # forward cap min(ceil(M/m), N-ti) and backward cap min(ceil(M/m), ti-1);
    # in 0-based those are min(ceil(M/m), N-1-ti) and min(ceil(M/m), ti), and
    # the forward lag hstar(1+m*tau) is 0-based hstar[m*tau] (the old port
    # used hstar[m*tau_pos - 1], which wrapped to the last element at
    # tau_pos=0 and shifted every forward lag by one).
    for m in range(1, int(np.round(n_fbins / 2.0))):
        factor = np.sqrt(m / (tbp * n_fbins))
        for icol in range(tcol):
            ti = int(timestamps[icol])
            cap_pos = int(min(np.ceil(M / m), xrow - 1 - ti))
            cap_neg = int(min(np.ceil(M / m), ti))
            tau_pos = np.arange(0, cap_pos + 1)
            tau_neg = np.arange(1, cap_neg + 1)
            # positive frequencies (0-based row m == MATLAB row 1+m)
            tfr[m, icol] = np.dot(hstar[m * tau_pos], signal[ti + tau_pos])
            tf2[m, icol] = np.dot(thstar[m * tau_pos], signal[ti + tau_pos])
            if tau_neg.size:
                tfr[m, icol] += np.dot(np.conj(hstar[m * tau_neg]),
                                       signal[ti - tau_neg])
                tf2[m, icol] -= np.dot(np.conj(thstar[m * tau_neg]),
                                       signal[ti - tau_neg])
            # negative frequencies (0-based row N-m == MATLAB row N+1-m)
            tfr[n_fbins - m, icol] = np.dot(np.conj(hstar[m * tau_pos]),
                                            signal[ti + tau_pos])
            tf2[n_fbins - m, icol] = np.dot(np.conj(thstar[m * tau_pos]),
                                            signal[ti + tau_pos])
            if tau_neg.size:
                tfr[n_fbins - m, icol] += np.dot(hstar[m * tau_neg],
                                                 signal[ti - tau_neg])
                tf2[n_fbins - m, icol] -= np.dot(thstar[m * tau_neg],
                                                 signal[ti - tau_neg])
        tfr[m, :] *= factor
        tf2[m, :] *= factor / m
        tfr[n_fbins - m, :] *= factor
        tf2[n_fbins - m, :] *= factor / m

    m = int(np.round(n_fbins / 2.0))
    factor = np.sqrt(m / (tbp * n_fbins))
    for icol in range(tcol):
        ti = int(timestamps[icol])
        cap_pos = int(min(np.ceil(M / m), xrow - 1 - ti))
        cap_neg = int(min(np.ceil(M / m), ti))
        tau_pos = np.arange(0, cap_pos + 1)
        tau_neg = np.arange(1, cap_neg + 1)
        tfr[m, icol] = np.dot(hstar[m * tau_pos], signal[ti + tau_pos])
        tf2[m, icol] = np.dot(thstar[m * tau_pos], signal[ti + tau_pos])
        if tau_neg.size:
            tfr[m, icol] += np.dot(np.conj(hstar[m * tau_neg]),
                                   signal[ti - tau_neg])
            tf2[m, icol] -= np.dot(np.conj(thstar[m * tau_neg]),
                                   signal[ti - tau_neg])
    tfr[m, :] *= factor
    tf2[m, :] *= factor / m

    tfr, tf2 = tfr.ravel(), tf2.ravel()
    no_warn_mask = tfr != 0
    tf2[no_warn_mask] = tf2[no_warn_mask] / tfr[no_warn_mask]
    tfr = np.abs(tfr) ** 2
    tfr = tfr.reshape(n_fbins, tcol)
    tf2 = tf2.reshape(n_fbins, tcol).astype(complex)

    rtfr = np.zeros((n_fbins, tcol), dtype=complex)
    ex = np.mean(np.abs(signal) ** 2)
    threshold = ex * 1.0e-6
    factor = 2 * np.pi * n_fbins * (tbp ** 2)
    # Reassignment, transcribed 1:1 from MATLAB (lines 168-186) in 1-based
    # coordinates (the golden `hat` is stored 1-based), converting to 0-based
    # only for the rtfr index. The old port fed 0-based jcol/icol into the
    # 1-based m / icolhat / jcolhat formulas and then re-subtracted 1, which
    # corrupted both the reassignment targets and the stored hat vectors.
    R = np.round(n_fbins / 2.0)
    for icol in range(tcol):
        for jcol in range(n_fbins):
            if tfr[jcol, icol] > threshold:
                icol1 = icol + 1
                jcol1 = jcol + 1
                icolhat = int(np.clip(icol1 + np.round(np.real(tf2[jcol, icol] / dt)),
                                      1, tcol))
                m = (np.remainder(jcol1 + R - 2, n_fbins) - R + 1)
                jcolhat = jcol1 + np.round(np.imag((m ** 2) * tf2[jcol, icol] / factor))
                jcolhat = int((((jcolhat - 1) % n_fbins) + n_fbins) % n_fbins) + 1
                rtfr[jcolhat - 1, icolhat - 1] += tfr[jcol, icol]
                tf2[jcol, icol] = jcolhat + 1j * icolhat
            else:
                tf2[jcol, icol] = np.inf * (1 + 1j)
                rtfr[jcol, icol] += tfr[jcol, icol]

    return tfr, rtfr, tf2


def smoothed_pseudo_wigner_ville(signal, timestamps=None, n_fbins=None,
                                 twindow=None, fwindow=None):
    """smoothed_pseudo_wigner_ville

    :param signal:
    :param timestamps:
    :param n_fbins:
    :param twindow:
    :param fwindow:
    :type signal:
    :type timestamps:
    :type n_fbins:
    :type twindow:
    :type fwindow:

    :return:
    :rtype:
    """
    xrow = signal.shape[0]

    if timestamps is None:
        timestamps = np.arange(signal.shape[0])
    if n_fbins is None:
        n_fbins = signal.shape[0]
    if fwindow is None:
        hlength = np.floor(n_fbins / 4.0)
        hlength += 1 - (hlength % 2)
        fwindow = tftb_window(int(hlength), 'Hamming')
    elif fwindow.shape[0] % 2 == 0:
        raise ValueError('The smoothing window must have an odd length.')
    lh = (fwindow.shape[0] - 1) // 2

    if twindow is None:
        glength = np.floor(n_fbins / 10.0)
        glength += 1 - (glength % 2)
        twindow = tftb_window(int(glength), 'Hamming')
    elif twindow.shape[0] % 2 == 0:
        raise ValueError('The smoothing window must have an odd length.')
    lg = (twindow.shape[0] - 1) // 2

    tcol = timestamps.shape[0]
    deltat = timestamps[1:] - timestamps[:-1]
    if deltat.min() != deltat.max():
        raise ValueError("Time instants must be regularly sampled.")
    else:
        dt = deltat.min()

    tfr = np.zeros((n_fbins, tcol), dtype=complex)
    tf2 = np.zeros((n_fbins, tcol), dtype=complex)
    tf3 = np.zeros((n_fbins, tcol), dtype=complex)
    dh = derive_window(fwindow)

    for icol in range(tcol):
        ti = timestamps[icol]
        # MATLAB (1-based ti): taumax = min(ti+Lg-1, xrow-ti+Lg, round(N/2)-1, Lh)
        # 0-based:            taumax = min(ti+lg, xrow-1-ti+lg, round(N/2)-1, lh)
        taumax = min([ti + lg, xrow - 1 - ti + lg, np.round(n_fbins / 2.0) - 1,
                      lh])
        # tau = 0  (MATLAB lines 112-117)
        points = np.arange(-min([lg, xrow - 1 - ti]), min([lg, ti]) + 1)
        g2 = twindow[lg + points]
        g2 = g2 / g2.sum()
        tg2 = g2 * points
        xx = signal[ti - points] * np.conj(signal[ti - points])
        tfr[0, icol] = (g2 * xx).sum()
        tf2[0, icol] = (tg2 * xx).sum()
        tf3[0, icol] = dh[lh] * tfr[0, icol]

        for tau in range(1, int(taumax) + 1):
            points = np.arange(-min([lg, xrow - 1 - ti - tau]),
                               min([lg, ti - tau]) + 1)
            g2 = twindow[lg + points]
            g2 = g2 / g2.sum()
            tg2 = g2 * points
            xx = signal[ti + tau - points] * np.conj(signal[ti - tau - points])
            tfr[tau, icol] = (g2 * xx).sum() * fwindow[lh + tau]
            tf2[tau, icol] = fwindow[lh + tau] * (tg2 * xx).sum()
            tf3[tau, icol] = dh[lh + tau] * (g2 * xx).sum()
            tfr[n_fbins - tau, icol] = (g2 * np.conj(xx)).sum() * fwindow[lh - tau]
            tf2[n_fbins - tau, icol] = (tg2 * np.conj(xx)).sum() * fwindow[lh - tau]
            tf3[n_fbins - tau, icol] = dh[lh - tau] * (g2 * np.conj(xx)).sum()

    tfr = np.real(np.fft.fft(tfr, axis=0)).ravel()
    tf2 = np.real(np.fft.fft(tf2, axis=0)).ravel()
    tf3 = np.imag(np.fft.fft(tf3, axis=0)).ravel()

    no_warn_mask = tfr != 0
    tf2[no_warn_mask] = np.round(tf2[no_warn_mask] / tfr[no_warn_mask] / dt)
    tf3[no_warn_mask] = np.round(
        n_fbins * tf3[no_warn_mask] / tfr[no_warn_mask] / (2 * np.pi))
    tfr, tf2, tf3 = [x.reshape(n_fbins, tcol).astype(complex) for x in (tfr, tf2, tf3)]
    tf3 = np.real(tf3)

    ex = np.mean(np.abs(signal) ** 2)
    threshold = ex * 1.0e-6
    # MATLAB tfrrspwv: icolhat = icol - tf2 (time shift subtracted),
    # jcolhat = jcol - tf3, scatter to rtfr(jcolhat, icolhat). The core's
    # 2-axis mode is icolhat = icol + time_shift, so pass time_shift = -tf2.
    rtfr, hat = _reassign_scatter(tfr, -tf2, tf3, threshold, n_fbins, tcol)
    return tfr, rtfr, hat


def spectrogram(signal, time_samples=None, n_fbins=None, window=None):
    """Compute the spectrogram and reassigned spectrogram.

    :param signal: signal to be analzsed
    :param time_samples: time instants (default: np.arange(len(signal)))
    :param n_fbins: number of frequency bins (default: len(signal))
    :param window: frequency smoothing window (default: Hamming with \
        size=len(signal)/4)
    :type signal: array-like
    :type time_samples: array-like
    :type n_fbins: int
    :type window: array-like

    :return: spectrogram, reassigned specstrogram and matrix of reassignment \
        vectors
    :rtype: tuple(array-like)
    """
    if time_samples is None:
        time_samples = np.arange(signal.shape[0])
    elif np.unique(np.diff(time_samples)).shape[0] > 1:
        raise ValueError('Time instants must be regularly sampled.')
    if n_fbins is None:
        n_fbins = signal.shape[0]
    if window is None:
        wlength = int(np.floor(signal.shape[0] / 4.0))
        wlength += 1 - np.remainder(wlength, 2)
        window = tftb_window(wlength, 'Hamming')
    elif window.shape[0] % 2 == 0:
        raise ValueError('The smoothing window must have an odd length.')

    tfr = np.zeros((n_fbins, time_samples.shape[0]), dtype=complex)
    tf2 = np.zeros((n_fbins, time_samples.shape[0]), dtype=complex)
    tf3 = np.zeros((n_fbins, time_samples.shape[0]), dtype=complex)
    lh = (window.shape[0] - 1) // 2
    th = window * np.arange(-lh, lh + 1)
    dwin = derive_window(window)

    for icol in range(time_samples.shape[0]):
        ti = time_samples[icol]
        tau = np.arange(-np.min([np.round(n_fbins / 2) - 1, lh, ti]),
                        np.min([np.round(n_fbins / 2) - 1, lh,
                                signal.shape[0] - ti - 1]) + 1).astype(int)
        indices = np.remainder(n_fbins + tau, n_fbins)
        norm_h = np.linalg.norm(window[lh + tau], ord=2)
        tfr[indices, icol] = signal[ti + tau] * np.conj(window[lh + tau]) / norm_h
        tf2[indices, icol] = signal[ti + tau] * np.conj(th[lh + tau]) / norm_h
        tf3[indices, icol] = signal[ti + tau] * np.conj(dwin[lh + tau]) / norm_h

    tfr = np.fft.fft(tfr, axis=0).ravel()
    tf2 = np.fft.fft(tf2, axis=0).ravel()
    tf3 = np.fft.fft(tf3, axis=0).ravel()

    no_warn_mask = tfr != 0
    tf2[no_warn_mask] = np.round(np.real(tf2[no_warn_mask] / tfr[no_warn_mask]))
    tf3[no_warn_mask] = np.round(
        np.imag(n_fbins * tf3[no_warn_mask] / tfr[no_warn_mask] / (2 * np.pi)))

    tfr = np.abs(tfr) ** 2
    tfr = tfr.reshape(n_fbins, time_samples.shape[0])
    tf2 = tf2.reshape(n_fbins, time_samples.shape[0])
    tf3 = tf3.reshape(n_fbins, time_samples.shape[0])
    tf3 = np.real(tf3)

    ix = np.arange(time_samples.min(), time_samples.max() + 1) - 1
    rtfr, hat = _reassign_scatter(
        tfr, tf2, tf3, 1e-6 * np.mean(np.abs(signal[ix]) ** 2),
        n_fbins, time_samples.shape[0])
    return tfr, rtfr, hat


def reassigned_gabor(signal, time_samples=None, n_fbins=None, window_length=None,
                     k=0.001):
    """Compute the Gabor spectrogram and its reassigned version.

    MATLAB ``tfrrgab.m`` (F. Auger, 1994-1995). The analysis window is a
    Gaussian whose length is grown from ``window_length`` in steps of two
    (re-parameterizing the extremity value ``k`` each step) until the window
    tail satisfies ``h(end)*(Nh-1) <= 2*k``. The reassignment shifts each
    (time, frequency) bin by the local group delay (time) and by a Gaussian
    frequency correction (frequency), then scatters the energy.

    :param signal: signal to be analysed.
    :param time_samples: time instants (default ``np.arange(len(signal))``).
    :param n_fbins: number of frequency bins (default ``len(signal)``).
    :param window_length: initial (odd) Gaussian window length; default is the
        odd length ``floor(n_fbins/4)``.
    :param k: value at both extremities of the Gaussian window (default 0.001).

    :return: (tfr, rtfr, hat) — the Gabor spectrogram, the reassigned
        spectrogram, and the reassignment-vector matrix (``jcolhat + 1j*
        icolhat`` for reassigned bins, ``inf*(1+1j)`` for sub-threshold ones).
    :rtype: tuple(array-like)
    """
    if time_samples is None:
        time_samples = np.arange(signal.shape[0])
    if n_fbins is None:
        n_fbins = signal.shape[0]
    if window_length is None:
        hlength = int(np.floor(n_fbins / 4.0))
        hlength += 1 - np.remainder(hlength, 2)
        window_length = hlength
    window_length = int(window_length)
    if window_length % 2 == 0:
        raise ValueError('window_length must be odd')
    if window_length < 1:
        raise ValueError('window_length must be a positive scalar')
    if n_fbins <= 0:
        raise ValueError('n_fbins must be greater than zero')

    tcol = time_samples.shape[0]
    if tcol == 1:
        dt = 1.0
    else:
        deltat = np.diff(time_samples)
        if not np.allclose(deltat, deltat[0]):
            raise ValueError('Time instants must be regularly sampled.')
        dt = deltat[0]

    # grow the Gaussian window (MATLAB lines 84-92)
    nh = window_length
    k0 = k
    nh2 = nh - 2
    while True:
        nh2 += 2
        h = tftb_window(nh2, 'gauss', k ** ((nh2 - 1) ** 2 / (nh - 1) ** 2))
        if not (h[nh2 - 1] * (nh2 - 1) > 2 * k):
            break
    k = k ** ((nh2 - 1) ** 2 / (nh - 1) ** 2)
    nh = nh2
    lh = (nh - 1) // 2
    h = tftb_window(nh, 'gauss', k)
    th = h * np.arange(-lh, lh + 1)

    xrow = signal.shape[0]
    tfr = np.zeros((n_fbins, tcol), dtype=complex)
    tf2 = np.zeros((n_fbins, tcol), dtype=complex)
    for icol in range(tcol):
        ti = int(time_samples[icol])  # 0-based
        tau = np.arange(-min([int(np.floor(n_fbins / 2.0 + 0.5)) - 1, lh, ti]),
                        min([int(np.floor(n_fbins / 2.0 + 0.5)) - 1, lh,
                             xrow - 1 - ti]) + 1).astype(int)
        indices = np.remainder(n_fbins + tau, n_fbins)
        norm_h = np.linalg.norm(h[lh + tau], ord=2)
        tfr[indices, icol] = signal[ti + tau] * np.conj(h[lh + tau]) / norm_h
        tf2[indices, icol] = signal[ti + tau] * np.conj(th[lh + tau]) / norm_h

    tfr = np.fft.fft(tfr, axis=0).ravel()
    tf2 = np.fft.fft(tf2, axis=0).ravel()
    mask = tfr != 0.0
    tf3 = np.zeros(tfr.shape, dtype=float)
    tf3[mask] = np.round(np.imag(2 * np.log(k) * n_fbins * tf2[mask]
                                 / tfr[mask] / (2.0 * np.pi * lh ** 2)))
    tf2[mask] = np.round(np.real(tf2[mask] / tfr[mask] / dt))
    tfr = np.abs(tfr) ** 2
    tfr = tfr.reshape(n_fbins, tcol)
    tf2 = tf2.reshape(n_fbins, tcol)
    tf3 = tf3.reshape(n_fbins, tcol)

    rtfr, hat = _reassign_scatter(
        tfr, tf2, tf3, 1.0e-6 * np.mean(np.abs(signal) ** 2),
        n_fbins, tcol)
    return tfr, rtfr, hat


if __name__ == '__main__':
    from tftb.generators import fmlin
    import matplotlib.pyplot as plt
    ts = np.arange(128, step=2)
    sig = fmlin(128, 0.1, 0.4)[0]
    fwindow = kaiser(17, beta=3 * np.pi)
    _, rtfr, _ = pseudo_wigner_ville(sig, timestamps=ts, n_fbins=64,
                                     fwindow=fwindow)
    rtfr = np.abs(rtfr) ** 2
    threshold = np.amax(rtfr) * 0.05
    rtfr[rtfr <= threshold] = 0.0
    plt.imshow(rtfr[:64, :], aspect='auto', origin="lower",
               extent=[0, 128, 0, 0.5])
    plt.show()
