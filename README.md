# tftb — Time-Frequency Toolbox (Python port)

Complete Python port of the original MATLAB Time-Frequency Toolbox
(F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine), targeting numpy 2.x and
validated against the certified R2024a-compatible reference `tftb-0.3`.

> Status: all porting phases complete (0.3.0.dev0), pending the fresh-venv CI
> check. See `docs/porting-log.md` for per-function porting status and
> `docs/benchmarks.md` for performance.

## Install

```bash
python -m pip install -e .            # core
python -m pip install -e .[test]      # + test deps (pytest, h5py, scikit-image)
python -m pip install -e .[accel]     # + numba / cupy acceleration
```

Requires Python ≥ 3.11, numpy ≥ 2.1.

## Quickstart

```python
import numpy as np
from tftb.generators import fmlin, amgauss
from tftb.processing.cohen import WignerVilleDistribution

x = np.real(amgauss(256, 128, 60) * fmlin(256, 0.1, 0.3, 128)[0])
t, f, tfr = WignerVilleDistribution(x, sr=1000).run()
```

## Coverage vs the MATLAB toolbox

The port covers the MATLAB Time-Frequency Toolbox's core: signal generators,
the full Cohen-class (bilinear) family, reassignment, affine/wavelet transforms,
and the analysis/post-processing utilities. 88 public functions
(`tftb.generators`: 31, `tftb.processing`: 57), each validated against the
certified MATLAB R2024a reference via the golden-data pipeline — per-function
status and the golden outcomes are in [`docs/porting-log.md`](docs/porting-log.md).

| area | Python | MATLAB | notes |
|---|---|---|---|
| **Signal generators** | `fmlin`, `fmconst`, `fmpar`, `fmpower`, `fmsin`, `fmhyp`, `fmodany`, `amgauss`, `amexpos`, `amexpo2s`, `amrect`, `amtriang`, `anaask`, `anabpsk`, `anafsk`, `anaqpsk`, `anapulse`, `anastep`, `anasing`, `dopnoise`, `noisecg`, `noisecu`, `doppler`, `gdpower`, `klauder`, `mexhat`, `altes`, `atoms`, `griffitc`, `sigmerge`, `scale` | same names | 31 generators |
| **Cohen-class (bilinear)** | `WignerVilleDistribution`, `PseudoWignerVilleDistribution`, `smoothed_pseudo_wigner_ville`, `MargenauHillDistribution`, `Spectrogram`, `RihaczekDistribution`, `BornJordanDistribution`, `ChoiWilliamsDistribution`, `GeneralizedRectangularDistribution`, `ZaoAtlasMarksDistribution`, `ButterworthDistribution`, `MargenauHillSpectrogramDistribution`, `MinimumMeanCrossEntropyDistribution`, `ReducedInterferenceDistribution{Bessel,Binomial,Hann,Triangular}`, `StankovicDistribution` | `tfrwv`, `tfrpwv`, `tfrspwv`, `tfrmh`, `tfrsp`, `tfrri`, `tfrbj`, `tfrcw`, `tfrgrd`, `tfrzam`, `tfrbud`, `tfrmhs`, `tfrmmce`, `tfrdb*/tfrdh/tfrdt`, `tfrstank` | Numba-accelerated (Phase 5) |
| **Reassignment** | `reassigned_spectrogram`, `reassigned_gabor`, `reassigned_smoothed_pseudo_wigner_ville` | `tfrsp`/`tfrgabor`/`tfrspwv` (reassigned) | |
| **Affine / wavelet** | `ShortTimeFourierTransform`, `Scalogram`, `BertrandDistribution`, `DFlandrinDistribution`, `UnterbergerDistribution`, `cwt`, `cwt_mirrored`, `gaussn` | `tfrstft`, `tfrscalogram`, `tfrbertrand`, `tfrdflandrin`, `tfrunterberger`, `cwt` | |
| **Mellin / Zak** | `fmt`, `ifmt`, `zak`, `izak` | `tfmfmt`, `tfmifmt`, `tfmzak`, `tfmizak` | |
| **Parametric / AR** | `correlmx`, `kaytth`, `ifestar2`, `parafrep`, `tfrparam` | same names | |
| **Post-processing** | `ideal_tfr`, `renyi_information`, `plotifl` | ideal-TFR, Rényy entropy, `plotifl` | |
| **Analysis utilities** | `tfrideal`, `momftfr`, `momttfr`, `margtfr`, `holder`, `if2phase`, `midpoint`, `plotsid`, `d2statio`, `tffilter`, `tfristft`, `loctime`, `locfreq`, `inst_freq`, `group_delay` | same names | |

Not ported (MATLAB-only, non-core): the MATLAB figure/plotting layer, the
`tfm` (Mellin) GUI, and the `tfrhough` *image-based* Hough (the Python port
provides the numerical `hough` kernel used by the ridge analysis). See the
porting log for the full per-function map.

## Performance

The hot paths (Cohen-class tau loops, Stanković, Hough transform) are accelerated
with Numba and wired into the public API automatically (with silent fallback to
the pure-numpy path if Numba is unavailable). Typical measured speedups vs the
committed numpy baseline (GTX 1070 host, 12 threads; full tables and gate reviews
in [`docs/benchmarks.md`](docs/benchmarks.md)):

| function(s) | speedup | note |
|---|---|---|
| `hough` | **1325×** @ 256² | biggest single win (numpy rebuilt the θ-grid per pixel) |
| Cohen-class tau loops (`bj, cw, grd, bud, zam, ridb, ridbn, ridh, ridt`) | 33–79× @ 4096 | one shared `cohen_generic_numba` kernel |
| `spwv` | 32.9× @ 4096 | |
| `stank` | 23.1× @ 4096 | was O(N²), ~864 s @ 8192 on numpy |

FFT-bound functions (`wv`, `pwv`, `mh`, `pmh`, `mhs`, `stft`, `ri`, `cwt`, affine
family) stay on the already-fast numpy path — the speedup gate (≥3×) legitimately
ruled them out, and the demoted Numba kernels (`scalogram`, `renyi`) are kept in
the codebase (still identity-tested) so a faster machine can re-promote them.

**CuPy (opt-in, CUDA device, N ≥ 8192):** a second explicit API in
`tftb/_accel/cupy_kernels.py` — `spectrogram_cupy`, `cwt_cupy`, `spwv_cupy`.
On this GTX 1070 only `spectrogram_cupy` clears the gate (3.10×); the function
name is the selector (no env vars), there is no silent fallback, and the Numba
default path is untouched.

Install with `pip install -e .[accel]` for Numba/CuPy; the package is fully
functional without it (pure numpy).

License: GPL-2.0-or-later (inherited from the original MATLAB toolbox).
