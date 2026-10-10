addpath('D:/TOOLBOX/tftb-0.3/mfiles');
fid = fopen('C:/Users/MATTIA/AppData/Local/hermes/cache/scratch/analysis_probe.txt','w');
% ---- margtfr probe ----
N=128; sig = amgauss(N).*fmlin(N);
tfr = tfrwv(sig);
[margt,margf,E] = margtfr(tfr);
for i=1:N, fprintf(fid,'MT %d %.17e\n',i,margt(i)); end
for i=1:N, fprintf(fid,'MF %d %.17e\n',i,margf(i)); end
fprintf(fid,'ME %.17e\n',E);
% ---- d2statio probe (tfrspwv) ----
[d,f] = d2statio(sig);
for i=1:N, fprintf(fid,'D %d %.17e\n',i,d(i)); end
for i=1:N, fprintf(fid,'DF %d %.17e\n',i,f(i)); end
% ---- holder probe (tfrscalo) ----
h=0; t0=N/2; sig2 = anasing(N,t0,h);
[tfr2,t2,f2] = tfrscalo(sig2,1:N,16,0.01,0.49,2*N);
h0 = holder(tfr2,f2,1,length(f2),t0,0);
for i=1:length(f2), fprintf(fid,'HF %d %.17e\n',i,f2(i)); end
fprintf(fid,'HH %.17e\n',h0);
fclose(fid);
fprintf('wrote analysis_probe.txt\n');
