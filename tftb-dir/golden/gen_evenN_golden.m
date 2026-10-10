function gen_evenN_golden
%GEN_EVENN_GOLDEN  Even-N (power-of-2) goldens for plain WVD/PWV/SPWV.
% Determines whether the plain smoothed_pseudo_wigner_ville (cohen) is
% correct at even N (where it would have been implicitly exercised) or
% whether the odd-N golden failure is a pre-existing bug.
tftb_root = 'D:/TOOLBOX/tftb-0.3';
addpath(fullfile(tftb_root, 'mfiles'));
rng(0);
rawdir = 'D:/Source/tftb-dir/golden/raw';
if ~isfolder(rawdir), mkdir(rawdir); end
rtol = 1e-10;
for N = [128 256]
    x = amgauss(N, N/2) .* fmlin(N, 0.1, 0.4);
    [tfr, t, f] = tfrwv(x);
    save(fullfile(rawdir, sprintf('wv_%d.mat', N)), 'x', 'tfr', 't', 'f', 'rtol');
    [tfr, t, f] = tfrpwv(x);
    save(fullfile(rawdir, sprintf('pwv_%d.mat', N)), 'x', 'tfr', 't', 'f', 'rtol');
    [tfr, t, f] = tfrspwv(x);
    save(fullfile(rawdir, sprintf('spwv_%d.mat', N)), 'x', 'tfr', 't', 'f', 'rtol');
end
fprintf('gen_evenN_golden: wrote 6 even-N cases (wv/pwv/spwv x 128/256)\n');
end
