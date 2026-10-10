function y = lenfth(x)
% LENFTH Scratch shim: the certified tftb-0.3 contwtgnmir.m line 21 uses
% "1:lenfth(x)" but lenfth is not defined anywhere in the toolbox - an
% obvious typo for length. Supplied here (scratch path only) to enable the
% oracle; the intended behaviour is length(x).
y = length(x);
end
