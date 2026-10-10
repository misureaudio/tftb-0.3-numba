function y = fact2(n)
% FACT2 Double factorial: n!! = n*(n-2)*(n-4)*... (down to 1 or 2).
% Standard definition. Supplied in a scratch path because the certified
% tftb-0.3 toolbox references fact2 in gaussn.m but never ships it.
y = 1;
for k = n:-2:1
  y = y * k;
end
end
