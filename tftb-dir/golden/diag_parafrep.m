addpath('D:/TOOLBOX/tftb-0.3/mfiles');
rng(0);
noise=rand(1000,1); signal=filter([1 0 0],[1 1 1],noise);
Rx = correlmx(signal,4,'hermitian');
N=128;
fid=fopen('C:/Users/MATTIA/AppData/Local/hermes/cache/scratch/parafrep.txt','w');
for i=1:5, for j=1:5, fprintf(fid, 'Rx %d %d %.17e %.17e\n', i,j, real(Rx(i,j)), imag(Rx(i,j))); end
end
% AR: request A (the only method that assigns it).
[spec,freqs,A] = parafrep(Rx,128,'ar',2);
for i=1:N, fprintf(fid, 'S 1 %d %.17e\n', i, spec(i)); end
for i=1:length(A), fprintf(fid, 'A %d %.17e %.17e\n', i, real(A(i)), imag(A(i))); end
% others: request only 2 outputs (A unassigned -> R2024a errors on 3).
methods={'periodogram','capon','capnorm','lagunas','genlag'};
for m=1:5
  spec = parafrep(Rx,128,methods{m},2);
  for i=1:N, fprintf(fid, 'S %d %d %.17e\n', m+1, i, spec(i)); end
end
fclose(fid);
fprintf('wrote parafrep.txt\n');
