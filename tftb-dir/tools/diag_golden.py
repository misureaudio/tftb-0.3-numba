# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Diagnose gabor_1 (and re-verify rsp_1 off-by-one fix) vs MATLAB oracle."""
import numpy as np
from scipy.io import loadmat
from tftb.utils import divider, nearest_odd
from tftb.processing.linear import gabor as pgabor
from tftb.processing.reassigned import spectrogram as rspec

RAW = "golden/raw"

# ---- rsp_1 re-check (off-by-one fix) ----
d = loadmat(f"{RAW}/rsp_1.mat", squeeze_me=True)
x = np.asarray(d["x"], complex).ravel()
m_tfr = np.asarray(d["tfr"])
p_tfr, p_rtfr, _ = rspec(x)
print("=== rsp_1 tfr after off-by-one fix ===")
for s in (0, 1, -1):
    a, b = (p_tfr[:, s:], m_tfr[:, :m_tfr.shape[1]-s]) if s >= 0 else (p_tfr[:, :s], m_tfr[:, -s:])
    if a.size:
        print(f"  s={s:+d}: max_rel={np.abs(a-b).max()/m_tfr.max():.4f} corr={np.corrcoef(a.ravel(),b.ravel())[0,1]:.6f}")

# ---- gabor_1: correct window (odd(n_coeff) taps, both normalize) ----
print("\n=== gabor_1 window (correct length) ===")
N = x.shape[0]
n_coeff, _ = divider(N)
q, _ = divider(n_coeff)
win_len = nearest_odd(n_coeff)          # python window taps
win_py = np.exp(np.log(0.005)*np.linspace(-1,1,win_len)**2)
win_py = win_py/np.linalg.norm(win_py)  # python normalizes
win_ml = np.exp(np.log(0.005)*np.linspace(-1,1,win_len)**2)  # matlab odd(n_coeff) taps
win_ml = win_ml/np.linalg.norm(win_ml)  # matlab h=h/norm(h)
print(f"n_coeff={n_coeff} q={q} win_len={win_len}")
print(f"window max_rel = {np.abs(win_py-win_ml).max()/win_ml.max():.3e}")

# where is the tfr mismatch? per output
d = loadmat(f"{RAW}/gabor_1.mat", squeeze_me=True)
m_tfr = np.asarray(d["tfr"]); m_dgr = np.asarray(d["dgr"]); m_gam = np.asarray(d["gam"])
p_tfr, p_dgr, p_gam = pgabor(x)
print("\n=== gabor_1 shapes ===")
print("tfr", p_tfr.shape, m_tfr.shape, " dgr", p_dgr.shape, m_dgr.shape, " gam", p_gam.shape, m_gam.shape)
for nm, a, b in (("tfr",p_tfr,m_tfr),("dgr",p_dgr,m_dgr),("gam",p_gam,m_gam)):
    if a.shape==b.shape:
        print(f"{nm}: max_rel={np.abs(a-b).max()/max(b.max(),1e-300):.4f}  max_abs={np.abs(a-b).max():.3e}")
# gam is the dual window — check alignment
if p_gam.shape==m_gam.shape:
    print(f"gam: python[:6]={p_gam[:6].real}")
    print(f"gam: matlab[:6]={m_gam[:6].real}")
    for s in (-1,0,1):
        a = np.roll(p_gam, s)
        print(f"  gam roll s={s:+d}: max_rel={np.abs(a-m_gam).max()/max(m_gam.max(),1e-300):.4f}")
