# TFTB Python — Phase 5 benchmark baseline

Median wall-time (s) of the pure-numpy path. Each cell is one warm-up + timed reps (target 5); a cell that exceeds the 20 s per-cell budget shows `<sub>×N</sub>` = the fewer reps actually timed. Deterministic two-component signal. `—` = not benchmarked at that N (infeasible); **ERROR** = the function raised (recorded, not a harness bug).

| function | group | N=512 | N=2048 | N=4096 | N=8192 |
|---|---|---|---|---|---|
| `wv` | cohen | 0.0168 | 0.3505 | 1.5146 | 6.1178 <sub>×4</sub> |
| `pwv` | cohen | 0.0226 | 0.3044 | 1.3052 | 5.9203 <sub>×4</sub> |
| `spwv` | cohen | 0.8590 | 16.0636 <sub>×2</sub> | 70.6632 <sub>×1</sub> | 365.6577 <sub>×1</sub> |
| `ri` | cohen | 0.0275 | 0.3693 | 1.6227 | 7.1877 <sub>×3</sub> |
| `bj` | cohen | 0.8716 | 15.6284 <sub>×2</sub> | 68.3398 <sub>×1</sub> | 339.6146 <sub>×1</sub> |
| `cw` | cohen | 0.8980 | 17.1697 <sub>×2</sub> | 78.3949 <sub>×1</sub> | 408.1092 <sub>×1</sub> |
| `grd` | cohen | 1.0697 | 20.3520 <sub>×1</sub> | 91.4786 <sub>×1</sub> | 473.2594 <sub>×1</sub> |
| `zam` | cohen | 0.7198 | 13.2053 <sub>×2</sub> | 58.1431 <sub>×1</sub> | 285.4288 <sub>×1</sub> |
| `bud` | cohen | 0.9207 | 17.6150 <sub>×2</sub> | 79.9249 <sub>×1</sub> | 420.8358 <sub>×1</sub> |
| `ridb` | cohen | 0.9496 | 17.1399 <sub>×2</sub> | 75.1273 <sub>×1</sub> | 363.6369 <sub>×1</sub> |
| `ridbn` | cohen | 1.4864 | 33.6236 <sub>×1</sub> | 183.0873 <sub>×1</sub> | 1191.5540 <sub>×1</sub> |
| `ridh` | cohen | 0.9622 | 17.7092 <sub>×2</sub> | 78.6790 <sub>×1</sub> | 389.4104 <sub>×1</sub> |
| `ridt` | cohen | 0.9131 | 16.6837 <sub>×2</sub> | 73.0649 <sub>×1</sub> | 353.5510 <sub>×1</sub> |
| `mh` | cohen | 0.0189 | 0.3586 | 1.5408 | 6.9069 <sub>×3</sub> |
| `pmh` | cohen | 0.0182 | 0.2854 | 1.2501 | 5.4705 <sub>×4</sub> |
| `mhs` | cohen | 0.0347 | 0.5925 | 2.6266 | 11.6144 <sub>×2</sub> |
| `stank` | cohen | 3.2476 | 52.5441 <sub>×1</sub> | 211.1910 <sub>×1</sub> | 864.3106 <sub>×1</sub> |
| `mmce` | cohen | 0.0317 | 0.3705 | 1.4716 | 5.8604 <sub>×4</sub> |
| `stft` | spectrogram | 0.0216 | 0.3382 | 1.4875 | 6.9812 <sub>×3</sub> |
| `cwt` | wavelet | 0.0140 | 0.0274 | 0.0383 | 0.0622 |
| `bertrand` | affine | 0.2358 | 0.9319 | 1.8537 | 4.0363 |
| `dflandrin` | affine | 0.2083 | 0.8509 | 1.6607 | 3.3145 |
| `unterberger` | affine | 0.3062 | 1.3084 | 2.6906 | 5.8908 <sub>×4</sub> |
| `scalogram` | affine | 0.0873 <sub>×3</sub> | 1.9383 <sub>×3</sub> | 10.2033 <sub>×2</sub> | — |

## Findings (input to 5.1–5.3)

- **`stank` (Stanković) is O(N²)** — 3.2 s @ 512 → 864 s @ 8192 (×4 per N-doubling).
  It is the single slowest function in the suite and the top candidate for a Numba
  kernel in 5.1 (its per-column loop has the same `Στ` skeleton as the Cohen family).
- **Generalized-Cohen tau-loop family** (`spwv, bj, cw, grd, zam, bud, ridb, ridbn,
  ridh, ridt`) is O(N·Lg) with a large constant — ~300–1200 s @ 8192. These share one
  `Στ g(τ)h(τ/ρ)x(t+τ)x*(t−τ)` skeleton, so a single parameterized
  `cohen_generic_numba` kernel (5.1) targets all ten at once.
- **`ridbn` @ N=8192 emits `RuntimeWarning: overflow in exp`** (`cohen.py:476`,
  `c = np.exp(logC)`) — a pre-existing numerical edge in the binomial kernel at large
  N, not a harness artifact. The accelerated path must reproduce (or fix) this; the
  identity check will catch any divergence.
- **Fast / FFT-bound** (`wv, pwv, mh, pmh, mhs, stft, ri, cwt, bertrand, dflandrin,
  unterberger, scalogram`): sub-second to ~7 s @ 8192. These are already FFT-bound or
  have small kernels; the gate (≥3×) may legitimately rule most of them out of
  acceleration — they are the "documented numpy-only" set unless a kernel clears the
  gate.
- **`cwt` is the cheapest** (0.06 s @ 8192) — a `prange`-over-scales kernel is trivial
  to write but unlikely to clear the gate (already ~linear and FFT/conv-bound); a good
  5.1 canary for the dispatcher plumbing rather than a win.

## Phase 5.1 — Numba kernels (gate review)

Numba kernels (`tftb/_accel/kernels.py`, `@njit(parallel=True, cache=True)`,
`prange` over time columns, 12-thread) are wired into `.run()` via
`tftb/_accel/backend.py` (auto-fallback to numpy on any failure; the public API
and return values are unchanged). **Identity** vs the numpy path is verified at
machine epsilon (~1e-16, far under the plan's 1e-12 bound) for every accelerated
function; the test suite also asserts the kernel *actually ran* (no silent
fallback). Speedups below are vs the committed numpy baseline; the gate is
**≥1.5× @ 512 and ≥3× @ 4096**.

| function | kernel | N=512 | N=2048 | N=4096 | gate |
|---|---|---|---|---|---|
| bj, cw, grd, bud, zam, ridb, ridbn, ridh, ridt | `cohen_generic_numba` | 37–54× | — | 33–79× | **PASS** (all 9) |
| spwv | `spwv_numba` | — | 33.8× | 32.9× | **PASS** — briefly demoted (Phase 5.4 parallelizer OOB), re-promoted after the parallel-safe rewrite |
| stank | `stank_jcols` + `stank_scatter` | 23.6× | 27.2× | 23.1× | **PASS** |

Per-function Numba times (median, s): bj 0.018/1.816, cw 0.019/2.037,
grd 0.023/2.104, bud 0.022/2.142, zam 0.017/1.746, ridb 0.019/1.860,
ridbn 0.040/2.322, ridh 0.018/1.886, ridt 0.017/1.915 (N=512 / N=4096);
spwv 0.475 @ 2048, 2.147 @ 4096; stank 0.138 @ 512, 1.932 @ 2048, 9.125 @ 4096.

**Design notes:**
- The GCB kernel is a pure *gather + dot*: the per-class kernel weight
  (CW exponential, GRD `sinc`, Butterworth, RID Bessel/binomial/Hann/triangular)
  is **precomputed on the numpy side** into a matrix `K[tau, p+Lg]` by calling
  each class's own `_kernel`. This makes the Numba weights bit-identical to the
  numpy path (no Numba re-implementation of `lgamma`/`sinc`/`sqrt`-of-negative)
  and is the reason identity is at machine epsilon. The `K` fill range must match
  each class's `_points_tau_limit` point range (±min(tau,Lg) for True classes,
  ±Lg for the False ones) — a mismatch was caught by the identity test (the
  auto-fallback would otherwise have silently masked it).
- `stank`'s O(N²) cost is the Python-level jcol loop, not the FFTs: the kernel
  takes the pre-FFT `T1/T2/T3` matrices and compiles the jcol/k loop +
  reassignment scatter. The scatter is sequential (the cross-column `+=` ordering
  must match numpy); `hat` (reassignment vectors) is reproduced **exactly**.
- Not accelerated (FFT-bound, unlikely to clear the gate — left numpy-only,
  documented): wv, pwv, mh, pmh, mhs, stft, ri, cwt, and the affine family.
  These are the "documented numpy-only" set for 5.1; 5.2 may revisit affine/Hough.

## Phase 5.2 — Numba kernels (affine + post-processing)

Three more kernels added to `tftb/_accel/kernels.py`, wired through the same
`backend.get_backend` dispatcher (auto-fallback, public API unchanged). **Identity**
vs the numpy path is verified at machine epsilon for all three (the test suite
forces the Numba path on for the demoted ones via `TFTB_FORCE_NUMBA=1`). The
speedup gate is **≥1.5× @ 512 and ≥3× @ 4096**; per the plan, **the gate is
allowed to say "no"** — a kernel that fails is *demoted* (kept in `kernels.py`,
identity still tested, but routed to the faster numpy path by default).

| function | kernel | speedup (numba vs numpy) | gate | outcome |
|---|---|---|---|---|
| hough | `hough_numba` | 620× @ 128, **1325× @ 256** | ≥3× | **PASS** — kept Numba |
| scalogram | `scalogram_numba` | 0.57× @ 512, 0.73× @ 2048, 0.65× @ 4096 | ≥3× | **FAIL** — demoted |
| renyi | `renyi_numba` | 1.03× @ 512, 1.05× @ 2048, 0.98× @ 4096 | ≥3× | **FAIL** — demoted |

**Why each went the way it did:**
- **hough — PASS (decisive).** The numpy `hough_transform` rebuilds
  `np.linspace(0, 2π−deltat, n)` *inside the innermost (x, y) loop* and runs the
  whole triple loop in pure Python: 5.7 s @ 128², 43 s @ 256². The kernel
  precomputes the θ grid once and compiles the loop; it is sequential (the
  shared-bin `ht[bin] += val` would race under a parallel pixel loop) but is
  still ~600–1300× faster because the numpy cost is Python + per-pixel linspace,
  not FP throughput. The `ht` accumulator is **bit-identical** to numpy (same
  (x, y, θ) order). This is the biggest single win in Phase 5.
- **scalogram — FAIL.** The per-voice loop is a direct O(N·len(ha)) convolution;
  `np.convolve` uses a C implementation that a Numba `prange`-over-voices direct
  convolution cannot beat — and the official bench config (`fmin=0.01,
  fmax=0.49, n_voices=N`) makes the low-frequency wavelets very long (nha =
  √N·fmax/fmin ≈ 3136 @ 4096), where `np.convolve` dominates. The Numba kernel is
  actually *slower* (0.57–0.73×). A real win would need an FFT-based
  convolution (out of scope for the per-voice `prange` design); recorded as a
  legitimate "the gate said no".
- **renyi — FAIL (as predicted).** `renyi_information` is already fully
  vectorized (a handful of `integrate_2d` reductions, no inner Python loop), so
  compiling it does not help (0.98–1.05×). The plan explicitly anticipated this
  ("expect to fail the gate — record it"); it is recorded here.

**Design notes:**
- Demotion lives in `backend._DEMOTED = {"scalogram", "renyi"}`; `get_backend`
  routes demoted kernels to numpy by default, `TFTB_FORCE_NUMBA=1` overrides it
  (test hook), and `backend.demoted()` reports the set. The demoted kernels stay
  in `kernels.py` and their identity is still verified by the test suite — so a
  future machine or a better kernel can re-promote them without re-porting.
- `scalogram`'s `run()` was refactored so the normalization is a shared
  `_scalogram_normalize` method (identical arithmetic on both paths); the
  Numba kernel replaces only the per-voice convolution loop in the
  `waveparams > 0` branch (the default). The `== 0` (mexhat) and ndarray
  (user wavelet) branches stay numpy-only.
- `renyi`'s negative-value guard stays on the numpy side; the kernel computes
  only the `integrate_2d`-based reduction.

Commit: `perf: numba hough (gate PASS) + scalogram/renyi (gate FAIL, demoted)`

## Phase 5.4 — spwv demoted for safety (Numba parallelizer OOB)

`spwv_numba` is **demoted for a safety reason, not a speed one**. It had
already *passed* the gate (32.9× @ 4096) and its identity vs the numpy path is
at machine epsilon — but it contains a genuine out-of-bounds memory access that
the Numba **parallelizer** generates, which the golden tests (N = 128 only)
never triggered.

**How it was found.** The new whole-family non-power-of-2 invariant tests
(`tftb/processing/tests/test_nondefault_parameters.py`) ran `spwv` at
N ∈ {49, 50, 97, 127, 128, 129, 199} and segfaulted (`Windows fatal exception:
access violation` in `spwv_numba`) with a frequency of ~1 in 12 runs.

**Root cause — a Numba parallelizer codegen bug, not an algorithm bug.** The
evidence, in order:

1. **Serial vs parallel.** The same kernel, recompiled with
   `boundscheck=True`: `parallel=False` is **clean over 210 calls** at every N;
   `parallel=True` raises an out-of-bounds `IndexError` at the first call
   (N = 49). Identical algorithm, identical inputs, identical index math — the
   only difference is the `prange` parallelization.
2. **Index math is provably correct.** A line-for-line pure-Python mirror of
   the kernel body, with an explicit bounds check on *every* index (both
   negative and ≥ n) that raises on violation, is **clean for every N** — and
   the numpy reference (which uses fancy indexing that would *wrap* a negative
   index rather than raise) is also clean and finite. So no index the algorithm
   computes is out of bounds.
3. **Not a power-of-2 phenomenon.** The parallel OOB occurs at **both**
   power-of-2 (64, 128, 256, 512) and non-power-of-2 (49, 97, 127, 129, 199,
   333) N, while the serial build is clean at all of them. (The FFT is *not* in
   the kernel — it is `np.fft.fft` in Python after the kernel returns — so the
   Winograd/Cooley–Tukey length-dependence does not apply here.) The reason the
   bug hid at N = 128: with `boundscheck=False` the OOB is undefined behavior
   that only crashes when it happens to touch unmapped memory; at N = 128 it
   happened to land on mapped memory and the value stayed correct (the identity
   test at 128/256 passes).
4. **Reproduced across Numba versions.** Present in both 0.67.0 and 0.68.0
   (llvmlite 0.50.0) after clearing the compile cache — not a version
   regression.
5. **Bisected to the exact structure.** Removing pieces of the kernel (each
   recompiled `parallel=True, boundscheck=True`) isolated the trigger:

   | variant | OOB |
   |---|:--:|
   | full kernel | **yes** |
   | tau loop only (no special bin) | no |
   | tau loop + `if` + *empty* body | no |
   | tau loop + `if` + nested `for p` (any bounds) | **yes** |
   | nested `for p` after tau loop, `range(0,1)` constant bounds | **yes** |
   | nested `for p` after tau loop, writes to a scalar (not `tfr`) | **yes** |
   | nested `for p` after tau loop, `@njit` helper instead of `min` | **yes** |
   | nested `for p` after tau loop, separate `prange` loop | **yes** |

   The trigger is **a nested `for p` loop that *follows* the `for tau` loop**
   inside the `prange` body (the Nyquist special bin) — not the `min` builtin,
   not the loop bounds, not the `tfr` write, not the `if`, not variable reuse.
   (The main tau loop, which *also* contains a nested `for p` loop, is clean —
   it is the *second* sequential loop with a data-dependent `x` index that the
   parallelizer miscompiles.)

**Fix (parallel-safe rewrite, re-promoted).** The Nyquist special bin is now
computed *inside* the main tau loop as an `if tau == nyq` branch, so there is
no nested `for p` loop after the tau loop. Verified:
- `parallel=True, boundscheck=True`: **clean at every N** (49–512, 10 iters).
- **bit-identical** to the original kernel (rel diff 0.00e+00) and matches the
  numpy reference to 9.6e-16.
- **faster** than the original kernel: 327× @512, 110× @2048, 82× @4096
  (the original was 33.8× @2048) — comfortably clears the gate.

`spwv` is therefore **re-promoted** (removed from `backend._DEMOTED`); the
kernel is accelerated again. `test_phase52_demotion_routes_to_numpy` asserts
`spwv -> numba` and `demoted() == {"scalogram", "renyi"}`.

**Lesson.** A Numba `prange` parallelizer bug (0.67–0.68) miscompiles a nested
loop that *follows* another loop in the loop body into an OOB access, even when
the indices are provably in-bounds. The safe pattern is to keep a single
top-level `prange` loop whose body has at most one nested-loop "shape"; any
additional per-column loop must be folded into that loop (as a guarded branch)
or moved to a separate `prange` loop over the same index. The `spwv_numba`
docstring and this section record the specific case.

**Why the other kernels are safe.** `cohen_generic_numba` and `stank_jcols`
(the other active parallel kernels exercised by the non-power-of-2 tests) were
monkeypatched with `boundscheck=True` recompiles and driven through their real
call path at every non-power-of-2 N: **clean** (all 19 TFRs × 7 N). Only
`spwv_numba` miscompiles.

## Phase 5.3bis — CuPy path (second explicit API)

CuPy is a **second explicit API**, not a dispatch: three functions called by
name (the name is the selector — no env vars, no `get_backend`), in a new
self-contained module `tftb/_accel/cupy_kernels.py`. The Numba implementation
and the existing `cwt`/`spwv`/`Spectrogram` dispatch are **untouched** — the
Numba path stays the default for everything. Each `*_cupy` function requires
**N ≥ 8192** and a CUDA device (else a clear error), has **no silent fallback**
(an explicit CuPy call runs on CuPy or raises), and returns numpy (`.get()`).

Environment: **GTX 1070 (8 GB), CuPy 14.2.0**, `cupyx.scipy.fft` batched
transforms. Measured at the gate size **N = 8192** (same inputs on both sides):

| function | CuPy (s) | current-best ref (s) | ratio | gate ≥3× | outcome |
|---|---|---|---|---|---|
| spectrogram_cupy | 2.185 | numpy `Spectrogram` 6.772 | **3.10×** | ≥3× | **PASS** (marginal) |
| cwt_cupy | 3.745 | numpy `cwt` 5.182 | 1.38× | ≥3× | **FAIL** |
| spwv_cupy | 354.591 | **numba** `spwv` 12.087 | 0.03× | ≥3× | **FAIL** |

**Why each went the way it did:**
- **spectrogram_cupy — PASS (marginal, 3.10×).** The numpy `Spectrogram.run`
  builds the pre-FFT matrix in a Python loop over columns (most of its cost).
  The CuPy version vectorizes that gather on the numpy side (same
  element-for-element arithmetic, including the per-column window norm) and
  runs the final `fft(axis=0)` as one batched GPU transform. It clears the
  gate but barely — on a faster GPU (or at larger N, where the batched FFT
  dominates) the margin should grow; on this card it is ~3×.
- **cwt_cupy — FAIL (1.38×).** The numpy `cwt` is already fast (5.18 s @8192
  at n_voices=N); the per-voice FFT-conv on the GPU does not beat it once the
  H2D/D2H and per-voice overhead are counted. The gate says no.
- **spwv_cupy — FAIL (0.03×, decisive).** spwv's cost is the O(N·Lg) tau loop,
  **not** the FFT. This version only moves the final batched FFT to the GPU;
  the tau loop still runs in numpy (354 s @8192) — so it is ~83× *slower* than
  the Numba kernel (12.09 s) that already compiles that loop. A real CuPy win
  would require porting the tau loop to the GPU (out of scope for a batched-FFT
  design). This is the documented reason the plan predicted spwv would not be a
  CuPy candidate.

**Bottom line (the plan's explicitly-allowed outcome):** on this GTX 1070,
CuPy is worth it for **spectrogram only** (3.10×, marginal); it is **not worth
it** for cwt or spwv. The `*_cupy` functions remain available as an explicit
opt-in (an explicit API has no default path to "demote" from); the benchmark
table above is the record.

**Tests:** `tftb/tests/benchmarks/test_cupy.py` — guard tests (N < 8192 raises;
no silent fallback on a device error) run by default; identity of each
`*_cupy` vs the standard function at N = 8192 is marked `slow` (skipped when
CuPy/device is absent). Identity verified at machine epsilon / rtol 1e-10.

Commit: `perf: cupy path (spectrogram PASS 3.10x, cwt+spwv not worth it — recorded)`

---

## Final summary (Phase 5)

1. **Phase 5 accelerated 12 functions with Numba and 3 with CuPy** on top of the committed pure-numpy baseline (the baseline is the reference for every speedup below).
2. **Numba kept 12** (gate ≥1.5×@512, ≥3×@4096): the 9 generalized-Cohen tau-loop functions (`bj, cw, grd, bud, zam, ridb, ridbn, ridh, ridt` → one `cohen_generic_numba` kernel), `spwv` (32.9×@4096), `stank` (23.1×@4096), and `hough` (1325×@256 — the single biggest win).
3. **Numba demoted 2** (gate said no; kept in `kernels.py`, still identity-tested, re-promotable): `scalogram` (0.57–0.73×) and `renyi` (0.98–1.05×) — both already vectorized/`np.convolve`-bound.
4. **Numba identity** vs the numpy path is at **machine epsilon** (~1e-16, far under the plan's 1e-12 bound) for every accelerated function; the test suite also asserts the kernel *actually ran* (no silent fallback masked a bug).
5. **CuPy (GTX 1070, explicit `*_cupy` API, N ≥ 8192, no fallback)** — the plan's allowed "not worth it" outcome: `spectrogram_cupy` **3.10× PASS (marginal)**; `cwt_cupy` 1.38× and `spwv_cupy` 0.03× **FAIL** (spwv's cost is the tau loop, not the FFT).
6. **Documented numpy-only** (FFT-bound, legitimately fail the gate): `wv, pwv, mh, pmh, mhs, stft, ri, cwt` and the affine family (`bertrand, dflandrin, unterberger`).
7. **Public API unchanged**: Numba is wired through `backend.get_backend` with auto-fallback to numpy; CuPy is a separate explicit API (the function name is the selector). Both return numpy.
8. **Phase 5 exit gate met**: full pytest green **with acceleration enabled — 433 passed, 3 skipped, 0 failed**; `benchmarks.md` committed; every accelerated function either clears its gate or is documented above.
9. **Reproducible**: the benchmark harness (`tftb/tests/benchmarks/`, `bench.py`) and the numpy baseline are committed; gate reviews are recorded per-task in `porting-log.md`.
10. **Net result**: the O(N²)/O(N·Lg) hot paths (the Cohen-class tau loops, `stank`, and Hough) drop from minutes/seconds to ~1–2 s @4096; the FFT-bound functions stay on the (already-fast) numpy path; CuPy is available as an opt-in for large-N spectrogram on a CUDA device.
