function gen_pseudo_golden
%GEN_PSEUDO_GOLDEN  Goldens for plain PseudoPage (tfrppage) and
% PseudoMargenauHill (tfrpmh) — the plain variants had no golden before
% (only the reassigned variants were tested), so the scipy-vs-tftb_window
% default-window bug was never caught.
tftb_root = 'D:/TOOLBOX/tftb-0.3';
addpath(fullfile(tftb_root, 'mfiles'));
rng(0);
rawdir = 'D:/Source/tftb-dir/golden/raw';
if ~isfolder(rawdir), mkdir(rawdir); end
rtol = 1e-10;
for N = [128 127 121]
    x = amgauss(N, N/2) .* fmlin(N, 0.1, 0.4);
    [tfr, t, f] = tfrppage(x);
    save(fullfile(rawdir, sprintf('ppage_%d.mat', N)), 'x', 'tfr', 't', 'f', 'rtol');
    [tfr, t, f] = tfrpmh(x);
    save(fullfile(rawdir, sprintf('pmh_%d.mat', N)), 'x', 'tfr', 't', 'f', 'rtol');
end
fprintf('gen_pseudo_golden: wrote 6 cases (ppage/pmh x 128/127/121)\n');
end
