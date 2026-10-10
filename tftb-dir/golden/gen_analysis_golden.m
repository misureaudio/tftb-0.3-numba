% Task 4.5 analysis-utility golden cases (focused; the full generate_golden.m
% already wrote the pre-4.5 cases). Run from D:/Source/tftb-dir.
addpath('D:/TOOLBOX/tftb-0.3/mfiles');
rawdir = 'D:/Source/tftb-dir/golden/raw';
if ~exist(rawdir,'dir'), mkdir(rawdir); end
rtol = 1e-10;

% momftfr_1: time moments of the ideal TFR of a linear-chirp IF law.
N=128; [sig,ifl]=fmlin(N); x=sig; tfr=tfrideal(ifl);
[tm,D2]=momftfr(tfr);
save(fullfile(rawdir, 'momftfr_1.mat'), 'x', 'tfr', 'tm', 'D2', 'rtol');

% momttfr_1: frequency moments of the ideal TFR of a sinusoidal IF law.
[sig,ifl]=fmsin(N); x=sig; tfr=tfrideal(ifl);
[fm,B2]=momttfr(tfr,'type2');
save(fullfile(rawdir, 'momttfr_1.mat'), 'x', 'tfr', 'fm', 'B2', 'rtol');

% margtfr_1: time/frequency marginals + energy of the WVD.
sig = amgauss(N).*fmlin(N); x=sig; tfr = tfrwv(sig);
[margt,margf,E] = margtfr(tfr);
save(fullfile(rawdir, 'margtfr_1.mat'), 'x', 'tfr', 'margt', 'margf', 'E', 'rtol');

% holder_1: Holder exponent of an anasing singularity via tfrscalo.
N=256; t0=N/2; sig = anasing(N,t0,0); x=sig;
[tfr,t,f] = tfrscalo(sig,1:N,16,0.01,0.49,2*N);
h = holder(tfr,f,1,length(f),t0,0);
rtol = 1e-9;
save(fullfile(rawdir, 'holder_1.mat'), 'x', 'tfr', 'f', 't0', 'h', 'rtol');

% if2phase_1: phase from a smooth IF law (linear chirp).
% NOTE: MATLAB if2phase requires a ROW vector (integ rejects column inputs);
% fmlin returns a column, so transpose.
[sig,ifl]=fmlin(128); x=ifl.'; phi = if2phase(ifl.');
rtol = 1e-10;
save(fullfile(rawdir, 'if2phase_1.mat'), 'x', 'phi', 'rtol');

% d2statio_1: distance to stationarity of a windowed chirp.
sig = amgauss(128).*fmlin(128); x=sig;
[d,f] = d2statio(sig);
save(fullfile(rawdir, 'd2statio_1.mat'), 'x', 'd', 'f', 'rtol');

% tfristft_1: inverse STFT round-trip (two 0.2/0.4 tones, hamming(57)).
Nt=200; t=200+(-128:127); sig=[fmconst(200,0.2); fmconst(200,0.4)]; x=sig;
h=hamming(57); tfr=tfrstft(sig,t,256,h,0);
[sigsyn,tsyn]=tfristft(tfr,t,h,0); y=sigsyn;
rtol = 1e-10;
save(fullfile(rawdir, 'tfristft_1.mat'), 'x', 't', 'h', 'tfr', 'y', 'rtol');

fprintf('analysis goldens written to %s\n', rawdir);
