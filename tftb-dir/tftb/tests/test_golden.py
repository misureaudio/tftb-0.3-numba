# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Golden-reference comparison: Python port vs MATLAB tftb-0.3 oracle.

Each ``golden/raw/<case>.mat`` holds the shared input signal ``x`` plus the
MATLAB TFR output (``tfr``). The test loads the identical ``x``, runs the
corresponding Python computation, and asserts they agree at the stored
``rtol`` (1e-10 — deterministic FFT of the same input).

Run the MATLAB side first (batch):
    matlab -batch "addpath('D:/Source/tftb-dir/golden'); generate_golden"
"""
import os
import glob

import numpy as np
import pytest
from scipy.io import loadmat

from tftb.processing.cohen import (
    WignerVilleDistribution,
    PseudoWignerVilleDistribution,
    PageRepresentation,
    PseudoPageRepresentation,
    MargenauHillDistribution,
    PseudoMargenauHillDistribution,
    RihaczekDistribution,
    BornJordanDistribution,
    ChoiWilliamsDistribution,
    GeneralizedRectangularDistribution,
    ZaoAtlasMarksDistribution,
    ButterworthDistribution,
    ReducedInterferenceDistributionBessel,
    ReducedInterferenceDistributionBinomial,
    ReducedInterferenceDistributionHann,
    ReducedInterferenceDistributionTriangular,
    MargenauHillSpectrogramDistribution,
    smoothed_pseudo_wigner_ville as plain_spwv,
)
from tftb.processing.reassigned import spectrogram as reassigned_spectrogram
from tftb.processing.reassigned import reassigned_gabor
from tftb.processing.reassigned import morlet_scalogram
from tftb.processing.reassigned import (
    pseudo_wigner_ville as reassigned_pwv,
    pseudo_margenau_hill as reassigned_pmh,
    pseudo_page as reassigned_ppag,
    smoothed_pseudo_wigner_ville as reassigned_spwv,
)
from tftb.processing.affine import Scalogram, smoothed_pseudo_wigner
from tftb.processing.linear import gabor
from tftb.processing.linear import StankovicDistribution
from tftb.processing.cohen import MinimumMeanCrossEntropyDistribution
from tftb.processing import postprocessing as pproc
from tftb.processing.wavelet import gaussn, cwt, cwt_mirrored
from tftb.processing.mellin import fmt, ifmt
from tftb.processing.zak import zak
from tftb.processing.parametric import correlmx, kaytth, ifestar2, parafrep, tfrparam
from tftb.processing.analysis import (momftfr, momttfr, margtfr, holder,
                                      if2phase, d2statio, tfristft, tffilter)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "golden", "raw")
RAW_DIR = os.path.abspath(RAW_DIR)

# case-family -> dict of {matlab_varname: python_callable(x) -> array}.
# Every entry is compared against the identically-named MATLAB variable at rtol.
def _rsp(x):
    tfr, rtfr, hat = reassigned_spectrogram(x)
    return {"tfr": tfr, "rtfr": rtfr, "hat": hat}


def _gabor(x):
    tfr, dgr, gam = gabor(x)
    return {"tfr": tfr, "dgr": dgr, "gam": gam}


def _rgab(x):
    tfr, rtfr, hat = reassigned_gabor(x)
    return {"tfr": tfr, "rtfr": rtfr, "hat": hat}


def _freq_reassign(fn):
    """Frequency-only reassigned TFR: returns tfr, rtfr, and the 1-based
    frequency reassignment-vector (hat) matrix."""
    def _wrap(x):
        tfr, rtfr, hat = fn(x)
        return {"tfr": tfr, "rtfr": rtfr, "hat": hat}
    return _wrap


def _rspwv(x):
    # 2-axis reassigned smoothed pWV (complex hat: jcolhat + 1j*icolhat).
    tfr, rtfr, hat = reassigned_spwv(x)
    return {"tfr": tfr, "rtfr": rtfr, "hat": hat}


def _stan(x, data=None):
    # tfrrstan stores [stan, rtfr, hat] (no t/f). Task 2.9 validated the base
    # `stan`; Task 3.4 validates the reassigned rtfr/hat. The custom frequency
    # window G (stan_2) is read back from the .mat.
    G = np.asarray(data["G2"]).ravel() if (data is not None and "G2" in data) else None
    stan, ts, freqs, rtfr, hat = StankovicDistribution(x, fsmooth=G).run()
    return {"stan": stan, "rtfr": rtfr, "hat": hat}


def _mmce(x, data=None):
    # tfrmmce: windows are the columns of H (saved in the .mat).
    H = np.asarray(data["H"])
    return {"tfr": MinimumMeanCrossEntropyDistribution(x, H).run()[0]}


# ---- Task 3.2 post-processing mappings ----
# These take (tfr, hat, t) — read back from the .mat — and validate the
# post-processing function in isolation against the stored MATLAB output.
def _fried(x, data=None):
    tfr = np.asarray(data["tfr"], dtype=complex)
    hat = np.asarray(data["hat"], dtype=complex)
    return {"tifd": pproc.friedman_density(tfr, hat)}


# ridges mcode (stored in the .mat): 1 -> 'rsp', 2 -> 'rpwv'.
_RIDGES_METHOD = {1: "rsp", 2: "rpwv"}


def _ridg(x, data=None):
    tfr = np.asarray(data["tfr"], dtype=complex)
    hat = np.asarray(data["hat"], dtype=complex)
    t = np.asarray(data["t"]).ravel()
    mcode = int(np.asarray(data["mcode"]).ravel()[0])
    pt, pf = pproc.ridges(tfr, hat, t, method=_RIDGES_METHOD[mcode])
    return {"pointst": pt, "pointsf": pf}


def _htl(x, data=None):
    IM = np.asarray(data["IM"])
    # MATLAB htl(IM, 128, 128): M=N=128 samples along rho and theta.
    ht, rho, theta = pproc.hough_transform(IM, m=128, n=128)
    return {"ht": ht, "rho": rho, "theta": theta}


# ---- Task 3.3 energy-dot mappings ----
def _extr(x, data=None):
    B = np.asarray(data["B"])
    image2, nb_dots = pproc.imextrac(B)
    return {"Image2": image2, "NbDots": int(nb_dots)}


# ---- Task 4.1 continuous-wavelet-transform mappings ----
def _gaussn(x, data=None):
    # The wavelet itself: gaussn(0.1, 2).
    return {"g": gaussn(0.1, 2)}


def _cwt(x, data=None):
    # contwtgn(x, 0.01, 0.5, 64, 2).
    scalo, f, T, a, wt = cwt(x, fmin=0.01, fmax=0.5, n_voices=64, wave=2)
    return {"scalo": scalo, "f": f, "T": T, "a": a, "wt": wt}


def _cwtmp(x, data=None):
    # contwtgnmir(x, 0.01, 0.5, 64, 2).
    scalo, f, T, a, wt = cwt_mirrored(x, fmin=0.01, fmax=0.5, n_voices=64, wave=2)
    return {"scalo2": scalo, "wt2": wt}


# ---- Task 4.2 fast Mellin-transform mappings ----
def _fmt(x, data=None):
    # fmt(x, 0.01, 0.5, N): full Mellin transform (mellin + beta).  N is the
    # number of even voices: len(x) if even, len(x)+1 if odd (fmt_1: 128/128,
    # fmt_2: 121/122 — the odd time-centering path).
    n = x.shape[0]
    N = n if n % 2 == 0 else n + 1
    mellin, beta = fmt(x, fmin=0.01, fmax=0.5, N=N)
    return {"mellin": mellin, "beta": beta}


# ---- Task 4.3 Zak-transform mappings ----
def _zak(x, data=None):
    # zak(x) with the default divider factorization (zak_1: 256->16x16,
    # zak_2: 315->15x21).
    return {"dzt": zak(x)}


# ---- Task 4.4a correlation-matrix mappings ----
def _correlmx(x, data=None):
    # correlmx(x, p, alg) for all four algorithms (p read from the .mat).
    p = int(np.asarray(data["p"]).ravel()[0])
    return {
        "R_herm": correlmx(x, p, "hermitian"),
        "R_fbherm": correlmx(x, p, "fbhermitian"),
        "R_burg": correlmx(x, p, "burg"),
        "R_fbburg": correlmx(x, p, "fbburg"),
    }


# ---- Task 4.4b Kay-Tretter mappings ----
def _kaytth(x, data=None):
    # kaytth(16) and kaytth(33) (x is a placeholder; the kernel is
    # independent of the signal).
    return {"k16": kaytth(16), "k33": kaytth(33)}


# ---- Task 4.4c AR(2) Prony IF mappings ----
def _ifestar2(x, data=None):
    # ifestar2(x) with the default t = 4:N (1-based).
    fnorm, t, ratio = ifestar2(x)
    return {"fnorm": fnorm, "t": t, "ratio": ratio}


# ---- Task 4.4e sliding-window parametric TFR mappings ----
def _tfrparam(x, data=None):
    # tfrparam(x, t, N, p, L, 0, Rxtype, method) with t = 1:length(x).
    # The per-case params are stored in the .mat as N{suffix}/... .
    suffix = "1" if "N1" in data else "2"
    N = int(np.asarray(data[f"N{suffix}"]).ravel()[0])
    p = int(np.asarray(data[f"p{suffix}"]).ravel()[0])
    L = int(np.asarray(data[f"L{suffix}"]).ravel()[0])
    Rxtype = str(np.asarray(data[f"Rxtype{suffix}"]).ravel()[0])
    method = str(np.asarray(data[f"method{suffix}"]).ravel()[0])
    tfr, t, f = tfrparam(x, np.arange(1, x.size + 1), N, p, L, 0, Rxtype, method)
    return {"tfr": tfr, "t": t, "f": f}


# ---- Task 4.5 analysis-utility mappings ----
# Each reads the TFR (or input) saved in the .mat and applies the analysis
# function, so the golden validates the analysis math (not the TFR generator).
def _momftfr(x, data=None):
    tfr = np.asarray(data["tfr"], dtype=complex)
    tm, D2 = momftfr(tfr)
    return {"tm": tm, "D2": D2}


def _momttfr(x, data=None):
    tfr = np.asarray(data["tfr"], dtype=complex)
    fm, B2 = momttfr(tfr, "type2")
    return {"fm": fm, "B2": B2}


def _margtfr(x, data=None):
    tfr = np.asarray(data["tfr"], dtype=complex)
    margt, margf, E = margtfr(tfr)
    return {"margt": margt, "margf": margf, "E": E}


def _holder(x, data=None):
    tfr = np.asarray(data["tfr"], dtype=complex)
    f = np.asarray(data["f"], dtype=float).ravel()
    t0 = int(np.asarray(data["t0"]).ravel()[0])
    h = holder(tfr, f, 1, len(f), t0, 0)
    return {"h": h}


def _if2phase(x, data=None):
    # x is the IF law (saved as 'x' in the .mat).
    phi = if2phase(x)
    return {"phi": phi}


def _d2statio(x, data=None):
    d, f = d2statio(x)
    return {"d": d, "f": f}


def _tfristft(x, data=None):
    tfr = np.asarray(data["tfr"], dtype=complex)
    t = np.asarray(data["t"], dtype=int).ravel()
    h = np.asarray(data["h"], dtype=float).ravel()
    y, _ = tfristft(tfr, t, h)
    return {"y": y}


def _tffilter(x, data=None):
    # x is the input signal; H is the filter's WVD; t the time instants.
    H = np.asarray(data["H"], dtype=complex)
    t = np.asarray(data["t"], dtype=int).ravel()
    y = tffilter(H, x, t)
    return {"y": y}


# ---- Task 4.4d parametric frequency-representation mappings ----
def _parafrep(x, data=None):
    # parafrep(Rx, 128, method, 2) for all six methods (Rx read from the .mat).
    Rx = np.asarray(data["Rx"], dtype=complex)
    N = 128
    spec_ar, freqs, A = parafrep(Rx, N, "ar", 2)
    spec_per, _, _ = parafrep(Rx, N, "periodogram", 2)
    spec_capon, _, _ = parafrep(Rx, N, "capon", 2)
    spec_capnorm, _, _ = parafrep(Rx, N, "capnorm", 2)
    spec_lagunas, _, _ = parafrep(Rx, N, "lagunas", 2)
    spec_genlag, _, _ = parafrep(Rx, N, "genlag", 2)
    return {
        "spec_ar": spec_ar, "spec_per": spec_per, "spec_capon": spec_capon,
        "spec_capnorm": spec_capnorm, "spec_lagunas": spec_lagunas,
        "spec_genlag": spec_genlag, "A": A, "freqs": freqs,
    }


# ---- Task 5.x: reassigned Morlet scalogram (tfrrmsc) ----
def _rmsc(x, data=None):
    # morlet_scalogram(x, tbp=f0T). The port's `tbp` occupies the same formula
    # position as MATLAB's `f0T`, so pass the stored f0T explicitly (the two
    # defaults differ: MATLAB f0T=2.5, port tbp=0.25). Returns the same
    # (tfr, rtfr, hat) triple as MATLAB tfrrmsc; hat is the 1-based complex
    # reassignment vector with inf at sub-threshold bins.
    f0T = float(np.asarray(data["f0T"]).ravel()[0])
    tfr, rtfr, hat = morlet_scalogram(x, tbp=f0T)
    return {"tfr": tfr, "rtfr": rtfr, "hat": hat}


# ---- Task 5.x: smoothed pseudo affine Wigner (tfrspaw) ----
_KLABEL = {0: "bertrand", -1: "unterberger", 0.5: "d_flandrin", 2: "aspwv"}


def _spaw(x, data=None):
    # smoothed_pseudo_wigner(x, timestamps=1..N, K, nh0, fmin, fmax) — the
    # affine family. MATLAB tfrspaw(X, time, K, nh0, ng0, fmin, fmax, N, trace)
    # with time=1:N (1-based), ng0=0, N=[] (auto voices). K maps to the port's
    # string labels. Returns (tfr, t, f); t is 1-based, f the geometric axis.
    K = int(np.asarray(data["K"]).ravel()[0])
    nh0 = float(np.asarray(data["nh0"]).ravel()[0])
    fmin = float(np.asarray(data["fmin"]).ravel()[0])
    fmax = float(np.asarray(data["fmax"]).ravel()[0])
    tfr, t, f = smoothed_pseudo_wigner(
        x, timestamps=np.arange(1, x.size + 1), K=_KLABEL[K],
        nh0=nh0, fmin=fmin, fmax=fmax,
    )
    return {"tfr": tfr, "t": t, "f": f}


_FAMILIES = {
    "wv": lambda x: {"tfr": WignerVilleDistribution(x).run()[0]},
    "pwv": lambda x: {"tfr": PseudoWignerVilleDistribution(x).run()[0]},
    "spwv": lambda x: {"tfr": plain_spwv(x)},
    "ppage": lambda x: {"tfr": PseudoPageRepresentation(x).run()[0]},
    "pmh": lambda x: {"tfr": PseudoMargenauHillDistribution(x).run()[0]},
    "page": lambda x: {"tfr": PageRepresentation(x).run()[0]},
    "mh": lambda x: {"tfr": MargenauHillDistribution(x).run()[0]},
    "ri": lambda x: {"tfr": RihaczekDistribution(x).run()[0]},
    "bj": lambda x: {"tfr": BornJordanDistribution(x).run()[0]},
    "cw": lambda x: {"tfr": ChoiWilliamsDistribution(x).run()[0]},
    "grd": lambda x: {"tfr": GeneralizedRectangularDistribution(x).run()[0]},
    "zam": lambda x: {"tfr": ZaoAtlasMarksDistribution(x).run()[0]},
    "bud": lambda x: {"tfr": ButterworthDistribution(x).run()[0]},
    "ridb": lambda x: {"tfr": ReducedInterferenceDistributionBessel(x).run()[0]},
    "ridbn": lambda x: {"tfr": ReducedInterferenceDistributionBinomial(x).run()[0]},
    "ridh": lambda x: {"tfr": ReducedInterferenceDistributionHann(x).run()[0]},
    "ridt": lambda x: {"tfr": ReducedInterferenceDistributionTriangular(x).run()[0]},
    "mhs": lambda x: {"tfr": MargenauHillSpectrogramDistribution(x).run()[0]},
    "stan": _stan,
    "mmce": _mmce,
    "rsp": _rsp,
    "rgab": _rgab,
    "gabor": _gabor,
    "rpwv": _freq_reassign(reassigned_pwv),
    "rpmh": _freq_reassign(reassigned_pmh),
    "rppag": _freq_reassign(reassigned_ppag),
    "rspwv": _rspwv,
    "fried": _fried,
    "ridg": _ridg,
    "htl": _htl,
    "extr": _extr,
    "gaussn": _gaussn,
    "cwt": _cwt,
    "cwtmp": _cwtmp,
    "fmt": _fmt,
    "zak": _zak,
    "correlmx": _correlmx,
    "kaytth": _kaytth,
    "ifestar2": _ifestar2,
    "parafrep": _parafrep,
    "tfrparam": _tfrparam,
    "momftfr": _momftfr,
    "momttfr": _momttfr,
    "margtfr": _margtfr,
    "holder": _holder,
    "if2phase": _if2phase,
    "d2statio": _d2statio,
    "tfristft": _tfristft,
    "tffilter": _tffilter,
    "rmsc": _rmsc,
    "spaw": _spaw,
}


def _cases():
    if not os.path.isdir(RAW_DIR):
        return []
    return sorted(glob.glob(os.path.join(RAW_DIR, "*.mat")))


_CASES = _cases()
_IDS = [os.path.basename(p)[:-4] for p in _CASES]


@pytest.mark.skipif(
    not _CASES,
    reason="no golden .mat files in golden/raw/ — run generate_golden.m in MATLAB first",
)
@pytest.mark.parametrize("matpath", _CASES, ids=_IDS)
def test_golden_matches_matlab(matpath):
    data = loadmat(matpath, squeeze_me=True)
    x = np.asarray(data["x"], dtype=complex).ravel()
    rtol = float(np.asarray(data["rtol"]).ravel()[0])

    case = os.path.basename(matpath)[:-4]
    family = case.split("_")[0]
    assert family in _FAMILIES, f"no Python mapping for case family {family!r}"

    if family in ("stan", "mmce", "fried", "ridg", "htl", "extr",
                  "gaussn", "cwt", "cwtmp", "fmt", "correlmx", "parafrep",
                  "tfrparam", "momftfr", "momttfr", "margtfr", "holder",
                  "tfristft", "tffilter", "rmsc", "spaw"):
        py_out = _FAMILIES[family](x, data)
    else:
        py_out = _FAMILIES[family](x)
    # Compare every Python-produced array against its identically-named MATLAB variable.
    for var, py in py_out.items():
        assert var in data, f"case {case!r}: MATLAB .mat missing {var!r}"
        ref = np.asarray(data[var])
        py = np.asarray(py)
        assert py.shape == ref.shape, (
            f"{case}/{var}: shape mismatch python {py.shape} vs matlab {ref.shape}"
        )
        # inf-aware comparison: the reassignment-vector (hat) matrices contain
        # inf at sub-threshold bins. Compare the inf pattern exactly, then the
        # finite entries at rtol. (A naive allclose would take atol=inf*rtol and
        # pass vacuously for any array holding an inf.)
        py_c = np.asarray(py, dtype=complex)
        ref_c = np.asarray(ref, dtype=complex)
        assert np.array_equal(np.isinf(py_c), np.isinf(ref_c)), (
            f"{case}/{var}: inf pattern mismatch (sub-threshold bin set differs)"
        )
        fin = ~np.isinf(py_c)
        if fin.any():
            diff = np.abs(py_c[fin] - ref_c[fin])
            scale = max(float(np.abs(ref_c[fin]).max()), 1e-300)
            np.testing.assert_allclose(
                py_c[fin], ref_c[fin], rtol=rtol, atol=rtol * scale,
                err_msg=(
                    f"{case}/{var}: max_abs_diff={diff.max():.3e}, "
                    f"max_rel={diff.max() / scale:.3e}, scale={scale:.3e}"
                ),
            )
