function generate_golden
%GENERATE_GOLDEN  Produce golden-reference .mat files for the Python port.
%
% Design: each case saves the INPUT signal x together with the MATLAB TFR
% output. The Python side loads the identical x, runs its own computation,
% and compares against the stored MATLAB result. Sharing the input removes
% any generator-return mismatch (the source of the 0.2.0 gabor/scale bugs)
% so only the TFR computation is validated.
%
% Batch usage:
%   matlab -batch "addpath('D:/Source/tftb-dir/golden'); generate_golden"

tftb_root = 'D:/TOOLBOX/tftb-0.3';
addpath(fullfile(tftb_root, 'mfiles'));
rng(0);

rawdir = 'D:/Source/tftb-dir/golden/raw';
if ~isfolder(rawdir), mkdir(rawdir); end

N = 128;
rtol = 1e-10;   % deterministic FFT-of-same-input: expect ~1e-14

% ---- case wv_1 : Wigner-Ville, Gaussian-windowed linear chirp ----
x = amgauss(N, 60) .* fmlin(N, 0.1, 0.4);
[tfr, t, f] = tfrwv(x);
save(fullfile(rawdir, 'wv_1.mat'), 'x', 'tfr', 't', 'f', 'rtol');

% ---- case page_1 : Page representation ----
[tfr, t, f] = tfrpage(x);
save(fullfile(rawdir, 'page_1.mat'), 'x', 'tfr', 't', 'f', 'rtol');

% ---- case mh_1 : Margenau-Hill distribution ----
[tfr, t, f] = tfrmh(x);
save(fullfile(rawdir, 'mh_1.mat'), 'x', 'tfr', 't', 'f', 'rtol');

% ---- case ri_1 : Rihaczek distribution (complex; auto) ----
[tfr, t, f] = tfrri(x);
save(fullfile(rawdir, 'ri_1.mat'), 'x', 'tfr', 't', 'f', 'rtol');

% ---- case bj_1 : Born-Jordan distribution (tfrbj) ----
[tfr, t, f] = tfrbj(x);
save(fullfile(rawdir, 'bj_1.mat'), 'x', 'tfr', 't', 'f', 'rtol');

% ---- case cw_1 : Choi-Williams distribution (tfrcw, default sigma=1) ----
[tfr, t, f] = tfrcw(x);
save(fullfile(rawdir, 'cw_1.mat'), 'x', 'tfr', 't', 'f', 'rtol');

% ---- case grd_1 : generalized rectangular distribution (tfrgrd, defaults) ----
[tfr, t, f] = tfrgrd(x);
save(fullfile(rawdir, 'grd_1.mat'), 'x', 'tfr', 't', 'f', 'rtol');

% ---- case zam_1 : Zao-Atlas-Marks representation (tfrzam) ----
[tfr, t, f] = tfrzam(x);
save(fullfile(rawdir, 'zam_1.mat'), 'x', 'tfr', 't', 'f', 'rtol');

% ---- case mhs_1 : Margenau-Hill-Spectrogram distribution (tfrmhs) ----
[tfr, t, f] = tfrmhs(x);
save(fullfile(rawdir, 'mhs_1.mat'), 'x', 'tfr', 't', 'f', 'rtol');

% ---- case bud_1 : Butterworth distribution (tfrbud, default sigma=1) ----
[tfr, t, f] = tfrbud(x);
save(fullfile(rawdir, 'bud_1.mat'), 'x', 'tfr', 't', 'f', 'rtol');

% ---- reduced-interference family (tfrridb/ridbn/ridh/ridt) ----
[tfr, t, f] = tfrridb(x);
save(fullfile(rawdir, 'ridb_1.mat'), 'x', 'tfr', 't', 'f', 'rtol');
[tfr, t, f] = tfrridbn(x);
save(fullfile(rawdir, 'ridbn_1.mat'), 'x', 'tfr', 't', 'f', 'rtol');
[tfr, t, f] = tfrridh(x);
save(fullfile(rawdir, 'ridh_1.mat'), 'x', 'tfr', 't', 'f', 'rtol');
[tfr, t, f] = tfrridt(x);
save(fullfile(rawdir, 'ridt_1.mat'), 'x', 'tfr', 't', 'f', 'rtol');

% ---- mmce_1 : MMCE, even N=128, 3 Hamming windows (11/15/19) ----
H = zeros(19,3);
H(10+(-5:5),1) = tftb_window(11);
H(10+(-7:7),2) = tftb_window(15);
H(10+(-9:9),3) = tftb_window(19);
[tfr, t, f] = tfrmmce(x, H);
save(fullfile(rawdir, 'mmce_1.mat'), 'x', 'tfr', 't', 'f', 'H', 'rtol');

% ---- mmce_2 : MMCE, odd N=121, gauss/kaiser/parzen windows ----
xsave = x;
x = amgauss(121, 60) .* fmlin(121, 0.1, 0.4);
H = zeros(19,3);
H(10+(-6:6),1) = tftb_window(13,'gauss');
H(10+(-8:8),2) = tftb_window(17,'kaiser');
H(10+(-7:7),3) = tftb_window(15,'parzen');
[tfr, t, f] = tfrmmce(x, H);
save(fullfile(rawdir, 'mmce_2.mat'), 'x', 'tfr', 't', 'f', 'H', 'rtol');
x = xsave;

% ---- case rsp_1 : reassigned spectrogram (tfrrsp) ----
% default window: hamming of odd length floor(N/4)+1-rem(...). Same as python.
[tfr, rtfr, hat] = tfrrsp(x);
save(fullfile(rawdir, 'rsp_1.mat'), 'x', 'tfr', 'rtfr', 'hat', 't', 'f', 'rtol');

% ---- case stan_1 : Stankovic distribution (tfrrstan) ----
% tfrrstan returns [stan, rtfr, hat] (NO t/f). Task 2.9 validates the base
% `stan`; rtfr/hat (reassignment) are validated in Task 3.4.
[stan, rtfr, hat] = tfrrstan(x);
save(fullfile(rawdir, 'stan_1.mat'), 'x', 'stan', 'rtfr', 'hat', 'rtol');

% ---- case stan_2 : Stankovic with custom frequency window G (non-default) ----
% exercises the circular smoothing loop for k=1..LG with LG=4.
G2 = tftb_window(9, 'Hanning');
[stan, rtfr, hat] = tfrrstan(x, 1:N, N, G2);
save(fullfile(rawdir, 'stan_2.mat'), 'x', 'stan', 'rtfr', 'hat', 'G2', 'rtol');

% ---- case gabor_1 : Gabor representation (tfrgabor) ----
[tfr, dgr, gam] = tfrgabor(x);
save(fullfile(rawdir, 'gabor_1.mat'), 'x', 'tfr', 'dgr', 'gam', 'rtol');

% ---- case rgab_1 : reassigned Gabor spectrogram (tfrrgab), even N=128 ----
[tfr, rtfr, hat] = tfrrgab(x);
save(fullfile(rawdir, 'rgab_1.mat'), 'x', 'tfr', 'rtfr', 'hat', 't', 'f', 'rtol');

% ---- case rgab_2 : reassigned Gabor spectrogram, odd N=127 ----
xsave = x;
x = amgauss(127, 60) .* fmlin(127, 0.1, 0.4);
[tfr, rtfr, hat] = tfrrgab(x);
save(fullfile(rawdir, 'rgab_2.mat'), 'x', 'tfr', 'rtfr', 'hat', 't', 'f', 'rtol');
x = xsave;

% ---- reassigned frequency-only family (Task 3.1 DRY refactor coverage) ----
% tfrrpwv / tfrrpmh / tfrrppag / tfrrspwv all return [tfr, rtfr, hat] with a
% frequency-only reassignment (hat = 1-based freq bin, inf for sub-threshold).
[tfr, rtfr, hat] = tfrrpwv(x);
save(fullfile(rawdir, 'rpwv_1.mat'), 'x', 'tfr', 'rtfr', 'hat', 't', 'f', 'rtol');
[tfr, rtfr, hat] = tfrrpmh(x);
save(fullfile(rawdir, 'rpmh_1.mat'), 'x', 'tfr', 'rtfr', 'hat', 't', 'f', 'rtol');
[tfr, rtfr, hat] = tfrrppag(x);
save(fullfile(rawdir, 'rppag_1.mat'), 'x', 'tfr', 'rtfr', 'hat', 't', 'f', 'rtol');
[tfr, rtfr, hat] = tfrrspwv(x);
save(fullfile(rawdir, 'rspwv_1.mat'), 'x', 'tfr', 'rtfr', 'hat', 't', 'f', 'rtol');

% ---- Task 3.2: post-processing (friedman / ridges / htl) ----
t = 1:N;
% Friedman (method-independent): frequency-only hat from tfrrpwv.
[tfr, rtfr, hat] = tfrrpwv(x);
tifd = friedman(tfr, hat, t, 'tfrrpwv');
save(fullfile(rawdir, 'fried_1.mat'), 'x', 'tfr', 'hat', 'tifd', 'rtol');
% Friedman: 2-axis hat from tfrrsp.
[tfr, rtfr, hat] = tfrrsp(x);
tifd = friedman(tfr, hat, t, 'tfrrsp');
save(fullfile(rawdir, 'fried_2.mat'), 'x', 'tfr', 'hat', 'tifd', 'rtol');
% Ridges (2-axis 'rsp', pointsf = indices/Nf). mcode: 1=rsp.
[tfr, rtfr, hat] = tfrrsp(x);
[pointst, pointsf] = ridges(tfr, hat, t, 'tfrrsp');
mcode = 1;
save(fullfile(rawdir, 'ridg_1.mat'), 'x', 'tfr', 'hat', 't', ...
      'pointst', 'pointsf', 'mcode', 'rtol');
% Ridges (frequency-only 'rpwv', pointsf = indices/(2*Nf)). mcode: 2=rpwv.
[tfr, rtfr, hat] = tfrrpwv(x);
[pointst, pointsf] = ridges(tfr, hat, t, 'tfrrpwv');
mcode = 2;
save(fullfile(rawdir, 'ridg_2.mat'), 'x', 'tfr', 'hat', 't', ...
      'pointst', 'pointsf', 'mcode', 'rtol');
% Hough transform on a binary slant-line image (lineplot from tests/).
addpath(fullfile(tftb_root, 'tests'));
IM = lineplot(20, pi/4, 64, 64);
[ht, rho, theta] = htl(IM, 128, 128);
save(fullfile(rawdir, 'htl_1.mat'), 'x', 'IM', 'ht', 'rho', 'theta', 'rtol');

% ---- Task 3.3: energy dot extraction (tfrsurf / imextrac) ----
% imextrac on the docstring's binary image (5 rows x 12 cols) as DOUBLE —
% exercises the union-by-rename when two dots meet. (MATLAB's imextrac only
% works on double: tfrsurf passes a logical Binary, and assigning an integer
% label into a logical array collapses to true, so MATLAB's own tfrsurf hits
% its "should never happen" branch on real TFRs. The Python port casts to
% int, so energy_dots is validated by a Python reference test, not a golden.)
B = [1 0 0 0 0 0 1 0 0 0 1 0;
     1 1 0 0 0 1 1 0 0 0 1 1;
     1 0 0 0 1 0 1 0 0 0 0 1;
     0 0 0 0 1 1 1 0 0 0 0 0;
     0 0 0 0 0 0 0 0 0 0 0 0];
[Image2, NbDots] = imextrac(B);
save(fullfile(rawdir, 'extr_1.mat'), 'x', 'B', 'Image2', 'NbDots', 'rtol');

% ---- Task 4.1: continuous wavelet transform (contwtgn / contwtgnmir /
%      gaussn). NOTE: the certified toolbox is missing fact2 (used by gaussn)
%      and lenfth (contwtgnmir line 21, a typo for length); scratch shims
%      golden/fact2.m + golden/lenfth.m restore the intended behaviour so the
%      oracle can generate these goldens. The Python port is validated against
%      them. ----
% gaussn: the wavelet itself (exercises fact2 in the analytic energy).
g = gaussn(0.1, 2);
save(fullfile(rawdir, 'gaussn_1.mat'), 'x', 'g', 'rtol');
% contwtgn: full CWT (wt + scalogram + frequency/scale axes).
xw = amgauss(128,60).*fmlin(128,0.1,0.4);
[scalo,f,T,a,wt,ws] = contwtgn(xw, 0.01, 0.5, 64, 2);
save(fullfile(rawdir, 'cwt_1.mat'), 'x', 'scalo', 'f', 'T', 'a', 'wt', 'rtol');
% contwtgnmir: mirrored CWT (per-voice nbmir mirroring).
[scalo2,f2,T2,a2,wt2,ws2] = contwtgnmir(xw, 0.01, 0.5, 64, 2);
save(fullfile(rawdir, 'cwtmp_1.mat'), 'x', 'scalo', 'scalo2', 'f', 'T', 'a', 'wt', 'wt2', 'rtol');

% ---- Task 4.2: fast Mellin transform (fmt / ifmt) ----
% fmt: full Mellin transform (mellin + beta) of a windowed linear chirp.
x = amgauss(128).*fmlin(128,0.1,0.4);
[mellin,beta] = fmt(x, 0.01, 0.5, 128);
save(fullfile(rawdir, 'fmt_1.mat'), 'x', 'mellin', 'beta', 'rtol');
% fmt_2: ODD signal length (N=121, 122 even voices) — exercises the odd
% time-centering path  M=(Mt+rem(Mt,2))/2,  t=(1:Mt)-M-1  in fmt.
x = amgauss(121).*fmlin(121,0.1,0.4);
[mellin,beta] = fmt(x, 0.01, 0.5, 122);
save(fullfile(rawdir, 'fmt_2.mat'), 'x', 'mellin', 'beta', 'rtol');

% ---- Task 4.3: Zak transform (zak / izak) ----
% zak_1: default divider (N=256 -> 16x16) of a windowed linear chirp.
x = fmlin(256,0.1,0.4).*amgauss(256,128,128);
dzt = zak(x);
save(fullfile(rawdir, 'zak_1.mat'), 'x', 'dzt', 'rtol');
% zak_2: ODD signal length N=315 (=3^2*5*7), default divider -> 15x21.
x = fmlin(315,0.1,0.4).*amgauss(315,round(315/2),round(315/2));
dzt = zak(x);
save(fullfile(rawdir, 'zak_2.mat'), 'x', 'dzt', 'rtol');

% ---- Task 4.4a: correlation matrix (correlmx), all 4 algorithms ----
% correlmx_1: all four algorithms (hermitian/fbhermitian/burg/fbburg) on a
% noisy 0.1-cosine, p=2 (the docstring's AR(2)-IF example signal).
x = real(fmconst(64,0.1)) + 0.4*randn(64,1);
p = 2;
R_herm   = correlmx(x,p,'hermitian');
R_fbherm = correlmx(x,p,'fbhermitian');
R_burg   = correlmx(x,p,'burg');
R_fbburg = correlmx(x,p,'fbburg');
save(fullfile(rawdir, 'correlmx_1.mat'), 'x', 'p', 'R_herm', 'R_fbherm', 'R_burg', 'R_fbburg', 'rtol');

% ---- Task 4.4b: Kay-Tretter filter (kaytth) ----
% kaytth_1: the kernel at several lengths (the shipped kayttht.m is an empty
% stub, so the golden is the value reference and the properties are tested in
% test_parametric_kaytth.py).
k16 = kaytth(16);
k33 = kaytth(33);
save(fullfile(rawdir, 'kaytth_1.mat'), 'x', 'k16', 'k33', 'rtol');

% ---- Task 4.4c: AR(2) 4-point Prony instantaneous frequency (ifestar2) ----
% ifestar2_1: a linear chirp (0.05->0.45); the estimator recovers the IF law
% to ~1e-2 (approximate 2.5-point delay).
x = real(fmlin(235,0.05,0.45));
[fnorm,t,ratio] = ifestar2(x);
save(fullfile(rawdir, 'ifestar2_1.mat'), 'x', 'fnorm', 't', 'ratio', 'rtol');
% ifestar2_2: a constant 0.15 tone (exact recovery to ~sqrt(eps)).
x = real(fmconst(235,0.15));
[fnorm,t,ratio] = ifestar2(x);
save(fullfile(rawdir, 'ifestar2_2.mat'), 'x', 'fnorm', 't', 'ratio', 'rtol');

% ---- Task 4.4d: parametric frequency representation (parafrep), 6 methods ----
% parafrep_1: all six methods (ar/periodogram/capon/capnorm/lagunas/genlag)
% on the docstring's AR(1)-filtered noise correlation matrix (p=4). The AR
% spectrum has a ~1e8 peak, so the inf-aware rtol comparison (relative to the
% max magnitude) is what makes it pass at 1e-10.
noise=rand(1000,1); signal=filter([1 0 0],[1 1 1],noise);
Rx = correlmx(signal,4,'hermitian');
N=128;
[spec_ar,freqs,A] = parafrep(Rx,N,'ar',2);
spec_per = parafrep(Rx,N,'periodogram',2);
spec_capon = parafrep(Rx,N,'capon',2);
spec_capnorm = parafrep(Rx,N,'capnorm',2);
spec_lagunas = parafrep(Rx,N,'lagunas',2);
spec_genlag = parafrep(Rx,N,'genlag',2);
% genlag computes (Rx)^2, amplifying the condition number, so its cross-
% platform agreement is ~1e-9 relative (BLAS-level float non-determinism);
% the other five methods + A match to ~1e-14. rtol set to the looser bound.
rtol = 1e-9;
save(fullfile(rawdir, 'parafrep_1.mat'), 'x', 'Rx', 'spec_ar', 'spec_per', 'spec_capon', ...
     'spec_capnorm', 'spec_lagunas', 'spec_genlag', 'A', 'freqs', 'rtol');

% ---- Task 4.4e: sliding-window parametric TFR (tfrparam) ----
% tfrparam_1: 'ar' / 'fbhermitian', p=3, L=21 (the docstring example), a
% 128-sample slice of the docstring signal, 128 bins.
% tfrparam_2: 'periodogram' / 'fbhermitian', p=15, L=61 (the docstring
% example), same signal slice, 256 bins.
xtf = fmconst(128,0.1)+fmlin(128,0.15,0.30)+fmlin(128,0.2,0.45);
x = xtf;   % so 'x' in the .mat is this 128-sample signal (not a stale case var)
[tfr,t,f] = tfrparam(xtf,1:128,128,3,21,0,'fbhermitian','ar');
N1=128; p1=3; L1=21; Rxtype1='fbhermitian'; method1='ar';
% AR divides by |Z'A|^2, amplifying float noise to ~2e-9 relative (same floor
% as the single parafrep AR case); the looser rtol is the AR bound.
rtol = 1e-8;
save(fullfile(rawdir, 'tfrparam_1.mat'), 'x', 'tfr', 't', 'f', 'N1', 'p1', 'L1', 'Rxtype1', 'method1', 'rtol');

[tfr,t,f] = tfrparam(xtf,1:128,256,15,61,0,'fbhermitian','periodogram');
N2=256; p2=15; L2=61; Rxtype2='fbhermitian'; method2='periodogram';
rtol = 1e-10;   % periodogram is stable: the standard deterministic bound
save(fullfile(rawdir, 'tfrparam_2.mat'), 'x', 'tfr', 't', 'f', 'N2', 'p2', 'L2', 'Rxtype2', 'method2', 'rtol');

% ---- Task 4.5: analysis utilities (moments, marginals, holder, phase,
% stationarity, filtering) ----
% momftfr_1: time moments of the ideal TFR of a linear-chirp IF law.
N=128; [sig,ifl]=fmlin(N); tfr=tfrideal(ifl);
[tm,D2]=momftfr(tfr);
save(fullfile(rawdir, 'momftfr_1.mat'), 'x', 'tfr', 'tm', 'D2', 'rtol');

% momttfr_1: frequency moments of the ideal TFR of a sinusoidal IF law.
[sig,ifl]=fmsin(N); tfr=tfrideal(ifl);
[fm,B2]=momttfr(tfr,'type2');
save(fullfile(rawdir, 'momttfr_1.mat'), 'x', 'tfr', 'fm', 'B2', 'rtol');

% margtfr_1: time/frequency marginals + energy of the WVD.
sig = amgauss(N).*fmlin(N); tfr = tfrwv(sig);
[margt,margf,E] = margtfr(tfr);
save(fullfile(rawdir, 'margtfr_1.mat'), 'x', 'tfr', 'margt', 'margf', 'E', 'rtol');

% holder_1: Holder exponent of an anasing singularity via tfrscalo.
N=256; t0=N/2; sig = anasing(N,t0,0);
[tfr,t,f] = tfrscalo(sig,1:N,16,0.01,0.49,2*N);
h = holder(tfr,f,1,length(f),t0,0);
% Python holder vs MATLAB holder on the SAME affine TFR: deterministic.
rtol = 1e-9;
save(fullfile(rawdir, 'holder_1.mat'), 'x', 'tfr', 'f', 't0', 'h', 'rtol');

% if2phase_1: phase from a smooth IF law (linear chirp).
% NOTE: MATLAB if2phase requires a ROW vector (integ rejects column inputs);
% fmlin returns a column, so transpose.
[sig,ifl]=fmlin(N); phi = if2phase(ifl.');
x = ifl.';   % the input to if2phase is the IF law (row)
rtol = 1e-10;
save(fullfile(rawdir, 'if2phase_1.mat'), 'x', 'phi', 'rtol');

% d2statio_1: distance to stationarity of a windowed chirp.
sig = amgauss(N).*fmlin(N);
[d,f] = d2statio(sig);
save(fullfile(rawdir, 'd2statio_1.mat'), 'x', 'd', 'f', 'rtol');

% tffilter_1: 2-component -> 1-component TF filter (tffilter.m Example 1).
Nt=128; t=1:Nt;
sig = fmlin(Nt,0.05,0.3) + fmlin(Nt,0.2,0.45); x=sig;
Nf=128; freqs = 0.5*(0:Nf-1).'/Nf;
H = zeros(Nf,Nt);
for tloop=1:Nt,
    rate = 0.2*(tloop-1)/(Nt-1);
    H(:,tloop) = (0+rate < freqs) .* (freqs < 0.1+rate);
end
y = tffilter(H,sig,t,0);
save(fullfile(rawdir, 'tffilter_1.mat'), 'x', 'H', 't', 'y', 'rtol');

% tfristft_1: inverse STFT round-trip (two 0.2/0.4 tones, hamming(57)).
Nt=200; t=200+(-128:127); sig=[fmconst(200,0.2); fmconst(200,0.4)];
h=hamming(57); tfr=tfrstft(sig,t,256,h,0);
[sigsyn,tsyn]=tfristft(tfr,t,h,0); y=sigsyn;
rtol = 1e-10;
save(fullfile(rawdir, 'tfristft_1.mat'), 'x', 't', 'h', 'tfr', 'y', 'rtol');

% NOTE: cross-WV (tfrwv([x1 x2])) deferred — Python base ravels 2-D input,
% so the cross convention needs separate mapping (follow-up case).

files = dir(fullfile(rawdir, '*.mat'));
fprintf('generate_golden: wrote %d cases to %s\n', numel(files), rawdir);
end
