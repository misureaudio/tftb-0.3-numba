% Task 4.5g tffilter golden (2-component -> 1-component TF filter).
% Follows tffilter.m Example 1: two linear chirps, a time-varying bandpass
% H tracks the first chirp (0.05->0.3) and rejects the second (0.2->0.45).
% Deterministic (no randomness) -> rtol 1e-10.
addpath('D:/TOOLBOX/tftb-0.3/mfiles');
rawdir = 'D:/Source/tftb-dir/golden/raw';
if ~exist(rawdir,'dir'), mkdir(rawdir); end
rtol = 1e-10;

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
fprintf('tffilter golden written\n');
