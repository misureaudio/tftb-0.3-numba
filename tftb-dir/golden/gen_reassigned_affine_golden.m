function gen_reassigned_affine_golden
%GEN_REASSIGNED_AFFINE_GOLDEN  Golden references for the two "paramount"
%TFRs that had NO golden coverage at any length:
%  - tfrrmsc  (reassigned Morlet scalogram)  -> Python reassigned.morlet_scalogram
%  - tfrspaw  (smoothed pseudo affine Wigner)-> Python affine.smoothed_pseudo_wigner
%
% Design (matches generate_golden.m): save the INPUT signal x together with
% the MATLAB TFR output; the Python side loads the identical x, runs its own
% computation, and compares. f0T (tfrrmsc) is stored explicitly so the
% Python side passes the *same* parameter value (tbp) it is validated with,
% independent of either default.
%
% Batch usage:
%   matlab -batch "addpath('D:/Source/tftb-dir/golden'); gen_reassigned_affine_golden"

tftb_root = 'D:/TOOLBOX/tftb-0.3';
addpath(fullfile(tftb_root, 'mfiles'));
rng(0);

rawdir = 'D:/Source/tftb-dir/golden/raw';
if ~isfolder(rawdir), mkdir(rawdir); end

N = 128;
rtol = 1e-10;   % deterministic FFT-of-same-input: expect ~1e-14

x = amgauss(N, 60) .* fmlin(N, 0.1, 0.4);

% ---- rmsc_1 : tfrrmsc, DEFAULT f0T (=2.5), N=128 ----
% tfrrmsc returns [tfr, rtfr, hat]; hat = complex reassignment vectors,
% inf at sub-threshold bins (Python morlet_scalogram returns the same triple).
f0T = 2.5;
[tfr, rtfr, hat] = tfrrmsc(x, 1:N, N, f0T, 0);
save(fullfile(rawdir, 'rmsc_1.mat'), 'x', 'tfr', 'rtfr', 'hat', 'f0T', 'rtol');

% ---- rmsc_2 : tfrrmsc, NON-default f0T (=1.5), exercises the wavelet-width
%      path (M, hstar, factor all scale with f0T). ----
f0T = 1.5;
[tfr, rtfr, hat] = tfrrmsc(x, 1:N, N, f0T, 0);
save(fullfile(rawdir, 'rmsc_2.mat'), 'x', 'tfr', 'rtfr', 'hat', 'f0T', 'rtol');

% ---- spaw_1 : tfrspaw, K=0 (Bertrand), default nh0=sqrt(N), ng0=0,
%      auto n_voices (N=[] -> Ndflt). fmin/fmax fixed so no prompt. ----
% tfrspaw returns [tfr, t, f]; t = time instants, f = geometric freq axis.
K = 0;
nh0 = sqrt(N);
fmin = 0.05; fmax = 0.45;
[tfr, t, f] = tfrspaw(x, 1:N, K, nh0, 0, fmin, fmax, [], 0);
save(fullfile(rawdir, 'spaw_1.mat'), 'x', 'tfr', 't', 'f', 'K', 'nh0', 'fmin', 'fmax', 'rtol');

% ---- spaw_2 : tfrspaw, K=2 (affine smoothed pseudo Wigner-Ville),
%      NON-default nh0=16 (exercises the mu/T/n_voices recompute path). ----
K = 2;
nh0 = 16;
fmin = 0.05; fmax = 0.45;
[tfr, t, f] = tfrspaw(x, 1:N, K, nh0, 0, fmin, fmax, [], 0);
save(fullfile(rawdir, 'spaw_2.mat'), 'x', 'tfr', 't', 'f', 'K', 'nh0', 'fmin', 'fmax', 'rtol');

files = dir(fullfile(rawdir, 'rmsc_*.mat'));
files2 = dir(fullfile(rawdir, 'spaw_*.mat'));
fprintf('gen_reassigned_affine_golden: wrote %d rmsc + %d spaw cases to %s\n', ...
        numel(files), numel(files2), rawdir);
end
