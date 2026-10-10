% Remaining analysis goldens (if2phase, d2statio, tfristft).
addpath('D:/TOOLBOX/tftb-0.3/mfiles');
rawdir = 'D:/Source/tftb-dir/golden/raw';
rtol = 1e-10;

[sig,ifl]=fmlin(128); x=ifl.'; phi = if2phase(ifl.');
save(fullfile(rawdir, 'if2phase_1.mat'), 'x', 'phi', 'rtol');

sig = amgauss(128).*fmlin(128); x=sig;
[d,f] = d2statio(sig);
save(fullfile(rawdir, 'd2statio_1.mat'), 'x', 'd', 'f', 'rtol');

Nt=200; t=200+(-128:127); sig=[fmconst(200,0.2); fmconst(200,0.4)]; x=sig;
h=hamming(57); tfr=tfrstft(sig,t,256,h,0);
[sigsyn,tsyn]=tfristft(tfr,t,h,0); y=sigsyn;
save(fullfile(rawdir, 'tfristft_1.mat'), 'x', 't', 'h', 'tfr', 'y', 'rtol');

fprintf('remaining analysis goldens written\n');
