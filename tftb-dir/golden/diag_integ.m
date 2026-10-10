% Probe MATLAB integ / integ2d conventions (tftb builtins)
Y = [1 2 3; 4 5 6; 7 8 9];   % 3x3
x = [0; 1; 2];               % length 3
f = [10; 20; 30];
disp('integ(Y) ='); disp(integ(Y));
disp('integ(Y,[0 1 2]) ='); disp(integ(Y, [0 1 2]));
disp('integ(Yt,[10 20 30]) ='); disp(integ(Y.', [10 20 30]));
disp('integ2d(Y,[0 1 2],[10;20;30]) ='); disp(integ2d(Y, [0 1 2], [10;20;30]));
disp('integ2d(Y,[10;20;30],[0 1 2]) ='); disp(integ2d(Y, [10;20;30], [0 1 2]));
