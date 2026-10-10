function gen_oddN_golden
%GEN_ODDN_GOLDEN  Non-power-of-2 golden cases for the core WVD/PWV/SPWV TFRs.
%
% Fills the odd-N / non-power-of-2 value-verification gap. The existing
% goldens cover these TFRs at power-of-2 N (wv_1 N=128; rpwv_1 / rspwv_1
% validate the *base* tfr of PWV/SPWV at N=128 through reassignment), and
% the plain PseudoWignerVille / SmoothedPseudoWignerVille had no golden at
% any length.
%
% The non-power-of-2 regime is where the odd-N index math (Nyquist bin,
% taumax cap, default-window length) lives. Three lengths are generated:
%   N=127  (N mod 4 == 3): round(N/2) agrees between Python (banker's) and
%          MATLAB (half-away-from-zero) -> control that odd-N index math is
%          correct when the rounding does not diverge.
%   N=121  (N mod 4 == 1): round(60.5) diverges (Python 60, MATLAB 61) ->
%          the taumax cap and the Nyquist special bin differ; this is the
%          case that exposes the latent Python-round bug.
%   N=129  (N mod 4 == 1): a second N mod 4 == 1 length confirming the fix.
%
% Each case saves the shared input x plus the MATLAB TFR; the Python side
% loads the identical x, runs its own computation, and compares at rtol
% (1e-10: deterministic FFT of the same input).
%
% Batch usage:
%   matlab -batch "addpath('D:/Source/tftb-dir/golden'); gen_oddN_golden"

tftb_root = 'D:/TOOLBOX/tftb-0.3';
addpath(fullfile(tftb_root, 'mfiles'));
rng(0);

rawdir = 'D:/Source/tftb-dir/golden/raw';
if ~isfolder(rawdir), mkdir(rawdir); end
rtol = 1e-10;

for N = [127 121 129]
    % Gaussian-windowed linear chirp (same convention as wv_1 / the even-N
    % rpwv_1 / rspwv_1 base cases).
    x = amgauss(N, N/2) .* fmlin(N, 0.1, 0.4);

    [tfr, t, f] = tfrwv(x);
    save(fullfile(rawdir, sprintf('wv_%d.mat', N)), 'x', 'tfr', 't', 'f', 'rtol');

    [tfr, t, f] = tfrpwv(x);
    save(fullfile(rawdir, sprintf('pwv_%d.mat', N)), 'x', 'tfr', 't', 'f', 'rtol');

    [tfr, t, f] = tfrspwv(x);
    save(fullfile(rawdir, sprintf('spwv_%d.mat', N)), 'x', 'tfr', 't', 'f', 'rtol');
end

files = dir(fullfile(rawdir, 'wv_12*.mat'));
files = [files; dir(fullfile(rawdir, 'pwv_*.mat'))];
files = [files; dir(fullfile(rawdir, 'spwv_*.mat'))];
fprintf('gen_oddN_golden: wrote %d odd-N cases (wv/pwv/spwv x 127/121/129)\n', numel(files));
end
