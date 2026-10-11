# `tftb` — API Manual

Python port of the MATLAB Time-Frequency Toolbox (F. Auger, P. Flandrin, P.
Gonçalves, O. Lemoine). This manual documents the public API and — with the
emphasis you asked for — the **data types** (shape, dtype, and units) of every
input parameter and output result.

> **Notation.** `int`/`float`/`str`/`bool` are Python builtins. `1-D`, `2-D`
> refer to array dimensionality. "Normalized frequency" means cycles/sample
> in `[0, 0.5]` (0.5 = Nyquist). "Time" is in sample indices unless a
> function's signature takes an explicit sampling frequency. Every array is a
> `numpy.ndarray` unless stated otherwise.

---

## 1. The data types that flow through the API

Everything in `tftb` is built from a small set of array shapes. Naming them
once makes the per-function tables below short.

| Symbol | Shape | dtype | Meaning |
|---|---|---|---|
| **Signal** `x` | `(N,)` 1-D | `complex128` or `float64` | The signal to analyze. Complex (analytic) is the normal case; real inputs are accepted and (for most TFRs) treated as-is. |
| **Window** `g`, `h`, `w` | `(L,)` 1-D | `float64` | Smoothing/analysis window. Produced by `tftb_window`. `L` is usually odd. |
| **Time axis** `t`, `ts` | `(Nt,)` 1-D | `float64` | Time instants (sample indices). Defaults to `arange(N)`. |
| **Freq axis (Cohen/linear)** `f`, `freqs` | `(Nf,)` 1-D | `float64` | Normalized-frequency bins, `fftshift`-ordered: `[0, …, 0.5−, −0.5, …, −]`. Range `(−0.5, 0.5)`. |
| **Freq axis (affine/wavelet)** `f` | `(Nv,)` 1-D | `float64` | Geometrically-spaced positive frequencies in `[fmin, fmax]`. |
| **TFR matrix** `tfr` | `(Nf, Nt)` 2-D | `complex128` *or* `float64` | The time-frequency representation. **Rows = frequency, columns = time.** Complex for the complex Cohen class (Rihaczek); real for WVD/PWV/Margenau-Hill/Page/Born-Jordan/Choi-Williams/Butterworth/RI/Bertrand/etc. |
| **Reassignment vectors** `hat` | `(Nf, Nt)` 2-D | `complex128` | `jcol + 1j·icol` per bin; `inf·(1+1j)` for sub-threshold (un-reassigned) bins. |
| **IF-law** `iflaw` | `(N,)` 1-D | `float64` | Instantaneous-frequency law in `[0, 0.5]`. |
| **IF-law matrix** `iflaws` | `(N, C)` 2-D | `float64` | `C` instantaneous-frequency laws, one per column (e.g. multi-component signal). |
| **Scalar** — | — | `float` | Single value (energy, exponent, information, …). |

**Two conventions worth knowing up front**

- **Row/column orientation.** A TFR matrix is always `(freq, time)`. When you
  plot it, use `origin='lower'` and `extent=[t0, t1, f0, f1]`.
- **Frequency axis differs by class.** Cohen-class and linear TFRs use the
  full `fftshift`-ed `(−0.5, 0.5)` axis (`freqs`). Affine (wavelet) TFRs use a
  geometrically-spaced positive axis in `[fmin, fmax]`. The axis you get back
  in the return tuple is the one to pair with the matrix.

---

## 2. Quick start

```python
from tftb.generators import fmlin
from tftb.processing import WignerVilleDistribution, PseudoWignerVilleDistribution

sig = fmlin(256, 0.1, 0.4)[0]          # (256,) complex128 analytic signal

tfr, t, f = WignerVilleDistribution(sig).run()
#   tfr : (256, 256) float64   -- real Cohen-class TFR
#   t   : (256,)    float64    -- time = arange(256)
#   f   : (256,)    float64    -- fftshift-ordered freq bins in (-0.5, 0.5)
```

---

## 3. Signal generators — `tftb.generators`

All generators take `n_points : int` as their first argument and return
`numpy.ndarray` (or a tuple of arrays). Signals are **analytic complex**
unless noted. "IF-law" means instantaneous frequency in `[0, 0.5]`.

### 3.1 Amplitude modulations

| Function | Signature (key params) | Returns |
|---|---|---|
| `amexpos` | `(n_points, t0=None, spread=None, kind='bilateral')` | `ndarray (N,) float` — exponential amplitude modulator |
| `amexpo2s` | `(n_points, t0=None, T=None)` | `ndarray (N,) float` — two-sided exponential modulator |
| `amgauss` | `(n_points, t0=None, spread=None)` | `ndarray (N,) float` — Gaussian modulator centered at `t0` |
| `amrect` | `(n_points, t0=None, spread=None)` | `ndarray (N,) float` — rectangular modulator |
| `amtriang` | `(n_points, t0=None, spread=None)` | `ndarray (N,) float` — triangular modulator |

### 3.2 Frequency modulations

| Function | Signature (key params) | Returns |
|---|---|---|
| `fmconst` | `(n_points, fnorm=0.25, t0=None)` | `(signal (N,) complex, iflaw (N,) float)` — constant-frequency signal |
| `fmlin` | `(n_points, fnormi=0.0, fnormf=0.5, t0=None)` | `(signal (N,) complex, iflaw (N,) float)` |
| `fmpar` | `(n_points, coefficients)` | `(signal (N,) complex, iflaw (N,) float)`; `coefficients=(a0,a1,a2)` parabolic |
| `fmpower` | `(n_points, k, coefficients)` | `(signal (N,) complex, iflaw (N,) float)` — power-law FM |
| `fmsin` | `(n_points, fnormin, fnormax, period, t0, fnorm0, pm1)` | `(signal (N,) complex, iflaw (N,) float)` |
| `fmhyp` | `(n_points, p1, p2)` | `(signal (N,) complex, iflaw (N,) float)` — hyperbolic FM |
| `fmodany` | `(iflaw, t0=1)` | `ndarray (N,) complex` — arbitrary FM from a given `iflaw (N,)` |
| `gdpower` | `(n_points, degree=0.0, rate=1.0)` | `(signal (N,) complex, group_delay (N,) float, freq_bins (N,) float)` |

### 3.3 Analytic / communication signals

| Function | Signature (key params) | Returns |
|---|---|---|
| `anaask` | `(n_points, n_comp=None, f0=0.25)` | `(signal (N,) complex, amplitude_mod (N,) float)` |
| `anabpsk` | `(n_points, n_comp=None, f0=0.25)` | `(signal (N,) complex, amplitude_mod (N,) float)` |
| `anafsk` | `(n_points, n_comp=None, Nbf=4)` | `(signal (N,) complex, iflaw (N,) float)` |
| `anaqpsk` | `(n_points, n_comp=None, f0=0.25)` | `(signal (N,) complex, phase (N,) float)` |
| `anapulse` | `(n_points, ti=None)` | `ndarray (N,) complex` — analytic impulse |
| `anastep` | `(n_points, ti=None)` | `ndarray (N,) complex` — analytic step |
| `anasing` | `(n_points, t0=None, h=0.0)` | `ndarray (N,) complex` — Lipschitz singularity |

### 3.4 Noise, Doppler, wavelets, misc

| Function | Signature (key params) | Returns |
|---|---|---|
| `noisecg` | `(n_points, a1=None, a2=None)` | `ndarray (N,) complex` — analytic Gaussian noise, mean 0 var 1 |
| `noisecu` | `(n_points)` | `ndarray (N,) complex` — analytic uniform white noise |
| `dopnoise` | `(n_points, s_freq, f_target, distance, v_target, time_center=None, c=340)` | `(signal (N,) complex, iflaw (N,) float)`; unit energy |
| `doppler` | `(n_points, s_freq, f0, distance, v_target, t0=None, v_wave=340.0)` | `(fm (N,) complex, am (N,) float, iflaw (N,) float)` |
| `klauder` | `(n_points, attenuation=10.0, f0=0.2)` | `ndarray (N,) float` — Klauder wavelet (time) |
| `mexhat` | `(nu=0.05)` | `ndarray (L,) float` — Mexican-hat wavelet (time) |
| `altes` | `(n_points, fmin=0.05, fmax=0.5, alpha=300)` | `ndarray (N,) float` — Altes signal |
| `atoms` | `(n_points, coordinates)` | `ndarray (N,) complex` — sum of Gaussian atoms |
| `sigmerge` | `(x1, x2, ratio=0.0)` | `ndarray (N,) complex` — `x1`+`x2` at energy ratio (dB) |
| `scale` | `(X, a, fmin, fmax, N)` | `ndarray (N,)` — Mellin-scaled signal |
| `griffitc` | `(n_points=200, snr=25)` | `(signal (N,) complex, iflaws (N, 3) float)` — 3-component test signal |

---

## 4. Time-frequency representations — `tftb.processing`

### 4.1 Class-based TFRs (Cohen class & linear)

These are classes: **`tfr, t, f = ClassName(signal, …).run()`**.

**Common `__init__` parameters** (inherited from `BaseTFRepresentation`):

| Param | Type | Default | Meaning |
|---|---|---|---|
| `signal` | `ndarray (N,)` complex/float | *required* | Input signal (ravelled if higher-dim) |
| `timestamps` | `ndarray (Nt,)` float | `arange(N)` | Time instants |
| `n_fbins` | `int` | `N` | Number of frequency rows |
| `fwindow` | `ndarray (L,)` float | `tftb_window(n//4, 'Hamming')` | Frequency-smoothing window |
| `twindow` | `ndarray (L,)` float | *(where applicable)* | Time-smoothing window |
| `**kwargs` | — | — | Distribution-specific knobs (`sigma`, `rs`, `MoverN`, …) |

**Common `.run()` return** — a 3-tuple:

```
tfr : ndarray (n_fbins, Nt)   -- complex128 (Rihaczek) or float64 (all others)
t   : ndarray (Nt,)    float  -- = timestamps
f   : ndarray (n_fbins,) float -- fftshift-ordered bins in (-0.5, 0.5)
```

**The Cohen-class family:**

| Class | Extra params | `tfr` dtype | Notes |
|---|---|---|---|
| `WignerVilleDistribution` | — | `float64` | Real WVD |
| `PseudoWignerVilleDistribution` | — | `float64` | PWV (time-windowed WVD) |
| `Spectrogram` / `ShortTimeFourierTransform` | — | `float64` | STFT / spectrogram `|STFT|²` |
| `RihaczekDistribution` | — | **`complex128`** | Complex Cohen class; `real(tfr)` = Margenau-Hill |
| `MargenauHillDistribution` | — | `float64` | Real part of Rihaczek |
| `BornJordanDistribution` | `twindow`, `fwindow` | `float64` | Two-window smoothing |
| `ChoiWilliamsDistribution` | `sigma=1.0` | `float64` | CW kernel |
| `GeneralizedRectangularDistribution` | `rs=1.0, MoverN=1.0` | `float64` | GRD kernel |
| `ButterworthDistribution` | `sigma=1.0` | `float64` | Butterworth kernel |
| `ReducedInterferenceDistributionBessel` | — | `float64` | Bessel kernel |
| `ReducedInterferenceDistributionBinomial` | — | `float64` | Binomial kernel |
| `ReducedInterferenceDistributionHann` | — | `float64` | Hann kernel |
| `ReducedInterferenceDistributionTriangular` | — | `float64` | Triangular kernel |
| `ZaoAtlasMarksDistribution` | — | `float64` | ZAM |
| `MargenauHillSpectrogramDistribution` | — | `float64` | MH-spectrogram product |
| `MinimumMeanCrossEntropyDistribution` | `windows (L, K)` float matrix | `float64` | Columns are freq windows (each odd, zero-padded to common length) |
| `StankovicDistribution` | `fwindow`, `fsmooth` | `float64` | Linear (STFT-based) freq-smoothed; **5-tuple** return `(tfr, ts, freqs, rtfr, hat)` |

> All of the above also expose `.plot(ax=None, kind='cmap'|'contour', show=True,
> …) -> None` for visualization, and the attributes `.tfr`, `.ts`/`.timestamps`,
> `.freqs`, `.n_fbins`, `.signal`.
>
> **Also top-level but a base class** — `GeneralizedCohenBase(signal,
> timestamps=None, n_fbins=None, twindow=None, fwindow=None, **kwargs)` is the
> shared skeleton for Born-Jordan / ZAM / reduced-interference; not meant to be
> instantiated directly (its `.run()` returns the usual 3-tuple).
>
> **Submodule-only classes** — `from tftb.processing.cohen import …` (not in
> `__all__`, so not on `tftb.processing` itself). Same 3-tuple `(tfr, ts,
> freqs)` return, real `float64` `tfr`: `PageRepresentation` (Page),
> `PseudoPageRepresentation` (pseudo-Page), `PseudoMargenauHillDistribution`
> (pseudo-Margenau-Hill).

### 4.2 Class-based affine (wavelet) TFRs

**`tfr, t, f = ClassName(signal, fmin=None, fmax=None, n_voices=None, …).run()`**

| Class | Extra params | `.run()` return |
|---|---|---|
| `Scalogram` | `waveparams=None` | `(tfr (n_voices, Nt) complex, ts (Nt,) float, f (n_voices,) float, wt (n_voices, Nt) complex)` — `tfr = |wt|²` (normalized), `f` = geometric voices |
| `BertrandDistribution` | — | `(tfr (n_voices, Nt) float, t (Nt,) float, f (n_voices,) float)` |
| `DFlandrinDistribution` | — | `(tfr (n_voices, Nt) float, t (Nt,) float, f (n_voices,) float)` |
| `UnterbergerDistribution` | `form='A'` | `(tfr (n_voices, Nt) float, t (Nt,) float, f (n_voices,) float)` |

`fmin`, `fmax` are normalized frequencies in `(0, 0.5]`; `n_voices` is the
number of geometric frequency rows.

### 4.3 Function-based (re)assigned TFRs

**Top-level** (in `tftb.processing.__all__`):

| Function | Signature (key params) | Returns |
|---|---|---|
| `smoothed_pseudo_wigner_ville` | `(signal, timestamps=None, freq_bins=None, twindow=None, fwindow=None)` | `tfr (Nf, Nt) float` — **single** real array (SPWV, `cohen` module; no reassignment) |
| `reassigned_spectrogram` | `(signal, time_samples=None, n_fbins=None, window=None)` | `(tfr (Nf, Nt) float, rtfr (Nf, Nt) float, hat (Nf, Nt) complex)` |
| `reassigned_gabor` | `(signal, time_samples=None, n_fbins=None, window_length=None, k=0.001)` | `(tfr (Nf, Nt) float, rtfr (Nf, Nt) float, hat (Nf, Nt) complex)` |
| `reassigned_smoothed_pseudo_wigner_ville` | `(signal, timestamps=None, n_fbins=None, twindow=None, fwindow=None)` | `(tfr (Nf, Nt) float, rtfr (Nf, Nt) float, hat (Nf, Nt) complex)` |

**Submodule-only** — `from tftb.processing.reassigned import …` (not in
`__all__`, so not on `tftb.processing` itself). Same `(tfr, rtfr, hat)` shape
unless noted:

| Function | Signature (key params) | Returns |
|---|---|---|
| `pseudo_wigner_ville` | `(signal, timestamps=None, n_fbins=None, fwindow=None)` | `(tfr, rtfr, hat)` |
| `pseudo_margenau_hill` | `(signal, timestamps=None, n_fbins=None, fwindow=None)` | `(tfr, rtfr, hat)` |
| `pseudo_page` | `(signal, timestamps=None, n_fbins=None, fwindow=None)` | `(tfr, rtfr, hat)` |
| `spectrogram` | `(signal, time_samples=None, n_fbins=None, window=None)` | `(tfr, rtfr, hat)` |
| `morlet_scalogram` | `(signal, timestamps=None, n_fbins=None, tbp=0.25)` | `(tfr (Nf, Nt) float, rtfr (Nf, Nt) float, tf2 (Nf, Nt) complex)` |

`hat` is `jcol + 1j·icol` per reassigned bin; `inf·(1+1j)` marks bins below the
reassignment threshold (left where they started).

### 4.4 Continuous wavelet transform (functions)

| Function | Signature | Returns |
|---|---|---|
| `cwt` | `(signal, fmin=0.01, fmax=0.5, n_voices=None, wave=2)` | `(scalo (n_voices, Nt) float, f (n_voices,) float, T (Nt,) float, a (n_voices,) float, wt (n_voices, Nt) complex)` — `scalo = |wt|²`; `a` = scales |
| `cwt_mirrored` | same as `cwt` | same as `cwt` (boundary-mirrored) |
| `gaussn` | `(f0, n)` | `ndarray (2N−1,) float` — order-`n` Gaussian-derivative wavelet, L2-normalized |

`wave` selects the wavelet family (int); `fmin`/`fmax` normalized, `fmax > fmin`.

### 4.5 Mellin, Zak, parametric, and other transforms

| Function | Signature | Returns |
|---|---|---|
| `fmt` | `(x, fmin, fmax, N=None)` | `(mellin (N,) complex, beta (N,) float)` — log-frequency Mellin domain |
| `ifmt` | `(mellin, beta, M=None)` | `ndarray (M,) complex` — reconstructed analytic signal |
| `zak` | `(sig, N=None, M=None)` | `ndarray (N, M) complex` — discrete Zak transform (rows FFT-scaled by `1/√N`) |
| `izak` | `(dzt)` | `ndarray (N·M,) complex` — inverse Zak |
| `correlmx` | `(x, p, rxtype='fbhermitian')` | `ndarray (p+1, p+1) complex` — correlation matrix up to lag `p` |
| `kaytth` | `(length)` | `ndarray (length,) float` — Kay-Tretter kernel, sums to 1 |
| `ifestar2` | `(x, t=None)` | `(fnorm (K,) float, t (K,) float, rejratio (K,) float)` — AR(2) Prony IF estimate |
| `parafrep` | `(Rx, N=128, method='capon', q=2)` | `(spec (N,) float, freqs (N,) float, A (…))` — parametric spectrum from `Rx` |
| `tfrparam` | `(x, t=None, N=None, p=None, L=None, trace=0, Rxtype='fbhermitian', method='ar', q=2)` | `(tfr (N, Nt) float, t (Nt,) float, f (N,) float)` — sliding-window parametric TFR |

`method` for `parafrep`/`tfrparam` ∈ `{'ar','capon','music','espri','pmd','mle'}`-style
parametric estimators.

### 4.6 Analysis / post-processing functions

| Function | Signature | Returns |
|---|---|---|
| `loctime` | `(sig)` | `(tm float, T float)` — mean time center, time spreading |
| `locfreq` | `(signal)` | `(fm float, bw float)` — mean freq center, freq spreading |
| `inst_freq` | `(x, t=None, L=1)` | `(fnorm (K,) float, t (K,) float)` — IF at the instants `t` |
| `group_delay` | `(x, fnorm=None)` | `ndarray (K,) float` — group delay at `fnorm` |
| `tfrideal` | `(iflaws, t=None, N=None, trace=0)` | `(tfr (N, Nt) float, t (Nt,) float, f (N,) float)` — impulse TFR at IF laws |
| `ideal_tfr` | `(iflaws, timestamps=None, n_fbins=None)` | `(tfr (Nf, Nt) float, ts (Nt,) float, f (Nf,) float)` |
| `momftfr` | `(tfr, tmin=None, tmax=None, time=None)` | `(tm (M,) float, D2 (M,) float)` — per-freq-row 1st/2nd time moments |
| `momttfr` | `(tfr, method, fbmin=None, fbmax=None, freqs=None)` | `(fm (N,) float, B2 (N,) float)` — per-time-col 1st/2nd freq moments |
| `margtfr` | `(tfr, t=None, f=None)` | `(margt (N,) float, margf (M,) float, E float)` — time marginal, freq marginal, energy |
| `holder` | `(tfr, f, n1=None, n2=None, t=None, pl=0)` | `float` (scalar) or `ndarray (K,)` if `t` is a vector — Holder exponent |
| `if2phase` | `(iflaw)` | `ndarray (N,) float` — cumulative phase from IF law |
| `midpoint` | `(t1, f1, t2, f2, k)` | `(ti float, fi float)` — interference point for affine-class power-`k` TFR |
| `plotsid` | `(t, iflaws, k=2)` | `list` of `(ti, fi)` pairs (auto-terms then cross-terms) |
| `d2statio` | `(sig)` | `(d (N,) float, f (N,) float)` — distance to stationarity (PWV-based) |
| `tffilter` | `(tfr, x, t, trace=0)` | `ndarray (len(t),) complex` — TF-filtered signal |
| `tfristft` | `(tfr, t, h, trace=0)` | `(x (len(t),) complex, t (len(t),) float)` — inverse STFT / TF synthesis |
| `renyi_information` | `(tfr, timestamps=None, freq=None, alpha=3.0)` | `float` — Rényi information of order `alpha` |
| `plotifl` | `(time_instants, iflaws, signal=None, **kwargs)` | `None` — plotting only |

**Submodule-only** — `from tftb.processing.postprocessing import …` (not in
`__all__`, so not on `tftb.processing` itself):

| Function | Signature | Returns |
|---|---|---|
| `hough_transform` | `(image, m=None, n=None)` | `(ht (m, n) float, rho (m,) float, theta (n,) float)` |
| `imextrac` | `(image)` | `(image2 (H, W) float, nb_dots int)` |
| `energy_dots` | `(tfr, threshold=5, keep=10)` | `(tfr2 (H, W) float, ordered_surfaces (K,) float)` |
| `friedman_density` | `(tfr, re_mat, timestamps=None)` | `ndarray (H, W) float` — time-IFD |
| `ridges` | `(tfr, re_mat, timestamps=None, method='rsp')` | `(time_points (K,) float, freq_points (K,) float)` — ridge extraction |

`re_mat` (in `friedman_density`/`ridges`) is the reassignment-vector matrix
(`(Nf, Nt)` complex, as produced by the reassigned TFRs in §4.3).

---

## 5. Window generation

```python
tftb_window(n, name='Hamming', param=None, param2=None) -> ndarray (n,) float
```

| Param | Type | Meaning |
|---|---|---|
| `n` | `int` | Window length (positive) |
| `name` | `str` | `'Hamming'` (default), `'Hanning'`, `'Nuttall'`, `'Papoulis'`, `'Harris'`, `'Rect'`, `'Triang'`, `'Bartlett'`, `'BartHann'`, `'Blackman'`, `'Gauss'`, `'Parzen'`, `'Kaiser'`, `'Dolph'`, `'Hanna'`, `'Nutbess'`, `'Spline'`, `'Flattop'` |
| `param` | `float` | Optional (Gauss edge value, Kaiser scale, …) |
| `param2` | `float` | Optional (Nutbess `nu`, Spline order `p`) |

> **Faithfulness note.** The MATLAB default `'Hamming'` is
> `0.54 − 0.46·cos(2π·(1:n)/(n+1))` — **not** `scipy.signal.windows.hamming`
> (different index spacing and endpoints). Every windowed TFR in this port
> uses `tftb_window`, so pass `tftb_window(…)` as `fwindow`/`twindow`/`window`
> to stay consistent with the MATLAB golden references.

---

## 6. Utilities — `tftb.utils`

| Function | Signature | Returns |
|---|---|---|
| `is_linear` | `(x, decimals=5)` | `bool` — whether `x` is (numerically) linear |
| `izak` | `(x)` | `ndarray` — inverse Zak (back-compat re-export of `tftb.processing.izak`) |
| `nextpow2` | `(n)` | `int` — smallest `k` with `2**k ≥ n` |
| `divider` | `(N)` | `(i int, factor int)` — factorization helpers |
| `nearest_odd` | `(N)` | `int` — nearest odd integer ≥ 1 |
| `modulo` | `(x, N)` | `ndarray` — element-wise `x mod N` |

`integrate_2d(mat, x=None, y=None) -> float` (in `tftb.processing.utils`)
numerically integrates a 2-D array over its axes.

---

## 7. Conventions & gotchas

- **Complex vs real TFR.** Only `RihaczekDistribution` returns a complex
  matrix (the complex Cohen class); `real(Rihaczek)` equals `MargenauHill`.
  Every other class-based and function-based TFR returns a real `float64`
  matrix. Reassigned TFRs additionally return the complex `hat` matrix.
- **Frequency axis.** Cohen/linear TFRs → `fftshift`-ed `(−0.5, 0.5)` bins
  (`freqs`). Affine/wavelet TFRs and `cwt` → geometric positive voices in
  `[fmin, fmax]`. Always use the axis returned in the tuple.
- **`timestamps` / `t`** default to `arange(N)` (sample indices). Pass a
  different 1-D float array to use physical time.
- **Windows** must come from `tftb_window` (see §5) to match MATLAB; `fwindow`
  defaults to a Hamming of length `n_fbins // 4` (rounded to odd).
- **`n_fbins`** defaults to the signal length `N`; the returned `tfr` has
  `n_fbins` rows regardless.
- **IF laws** are normalized to `[0, 0.5]`; some generators
  (`fmpar`, `fmsin`, …) emit a `UserWarning` if the law leaves that band
  (undersampling / negative frequency).
- **Multi-component** inputs: `iflaws` is `(N, C)` (one IF law per column) in
  `tfrideal`/`ideal_tfr`/`plotsid`; `griffitc` returns such a matrix.

---

*Generated from the live `tftb` package (introspected signatures and return
statements). If a signature or return tuple has changed, the source of truth is
the `:return:`/`:rtype:` docstrings and the `return` statements in
`tftb/processing/*.py` and `tftb/generators/*.py`.*
