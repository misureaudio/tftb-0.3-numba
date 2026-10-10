# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Phase 5.1/5.2 backend dispatcher.

Single public API, backend-internal dispatch (plan §Phase 5 principle 1). The
numpy path is always the reference and the fallback; a Numba kernel is only
used when (a) numba is importable, (b) no failure has been recorded for that
kernel, and (c) the kernel has not been *demoted* (it failed the speedup gate
on this machine — the gate is allowed to say "no", and a demoted kernel keeps
its numpy path as the default). Any exception raised by a kernel is caught,
recorded, and falls back to the numpy path, so the accelerated path can never
make a previously-working function fail.

``get_backend(name, n)`` returns ``"numba"`` or ``"numpy"``. Callers use it as::

    from tftb._accel import backend
    if backend.get_backend("cohen_generic", n) == "numba":
        try:
            tfr = kernels.cohen_generic_numba(...)
        except Exception:
            backend.report_failure("cohen_generic")
            tfr = _numpy_path(...)

The gate (>=3x @ N=4096, >=1.5x @ N=512) and the identity check (rtol 1e-12)
are enforced by the test suite (``tftb/tests/benchmarks/test_kernels.py``), not
here — this module only routes.

``TFTB_FORCE_NUMBA=1`` overrides demotion (a test hook: it forces a demoted
kernel onto the Numba path so its identity / gate can still be measured).
``TFTB_DISABLE_NUMBA=1`` forces every kernel onto the numpy path.
"""

import os

__all__ = ["get_backend", "report_failure", "is_enabled", "available_kernels",
           "demoted"]

# Kernels that exist in kernels.py. A kernel is "available" only if numba
# imported cleanly AND it has not failed at runtime in this process.
_KNOWN = ("cohen_generic", "spwv", "stank", "hough", "scalogram", "renyi")
# Kernels that FAILED the speedup gate on this machine (Phase 5.2 review):
# kept in kernels.py (identity still verified) but routed to numpy by default.
# The gate is allowed to say "no" — a demoted kernel is a legitimate outcome,
# not a bug. See docs/benchmarks.md (Phase 5.2 gate review).
#
# "spwv" was temporarily demoted (Phase 5.4) when its Numba parallelizer OOB
# was found; the parallel-safe rewrite (Nyquist special bin merged into the
# main tau loop, see spwv_numba's docstring) fixed it, so it is re-promoted.
_DEMOTED = frozenset({"scalogram", "renyi"})
_failed = set()


def _numba_present():
    try:
        import numba  # noqa: F401
        return True
    except Exception:
        return False


def _force():
    """True if ``TFTB_FORCE_NUMBA=1`` (test hook: run demoted kernels anyway)."""
    return os.environ.get("TFTB_FORCE_NUMBA") == "1"


def is_enabled():
    """Whether Numba acceleration is active at all (env + importable)."""
    if os.environ.get("TFTB_DISABLE_NUMBA") == "1":
        return False
    return _numba_present()


def demoted():
    """Kernels routed to numpy because they failed the speedup gate."""
    return tuple(k for k in _KNOWN if k in _DEMOTED)


def available_kernels():
    """Kernels actually used by the default path (numba present, not failed,
    not disabled, not demoted)."""
    if not is_enabled():
        return ()
    return tuple(k for k in _KNOWN
                 if k not in _failed and k not in _DEMOTED)


def get_backend(name, n=None):
    """Return ``"numba"`` if ``name`` should run accelerated, else ``"numpy"``.

    ``n`` is the signal length (reserved for a future size-based cutoff); the
    current gate is per-kernel, so it is accepted but not yet used.
    """
    if name not in _KNOWN:
        return "numpy"
    if not is_enabled():
        return "numpy"
    if name in _failed:
        return "numpy"
    if name in _DEMOTED and not _force():
        return "numpy"
    return "numba"


def report_failure(name):
    """Mark ``name`` as failed for the rest of this process (auto-fallback)."""
    _failed.add(name)


def reset_failures():
    """Clear recorded failures (test helper)."""
    _failed.clear()
