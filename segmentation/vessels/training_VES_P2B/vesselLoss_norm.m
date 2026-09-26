function [loss, gradients, cePart, dicePart, wV] = vesselLoss_norm(net, X, T, W0)
% vesselLoss_norm — P3 loss (new file; vesselLoss_frozen.m untouched).
% Change vs frozen (recheck Change 1): vessel term normalized by FIXED W0
% (dataset-level bg/vessel ratio over the 16 train images, logged once,
% identical both arms) — background term byte-identical to frozen math.
% cePart = -mean(G.*(wV/W0).*logP + (1-G).*logN) + batch soft-Dice smooth=1.
% Per-iter wV returned for diagnosis (no longer in the scale path).
Y = forward(net, X);
P = Y(:, :, 2, :);
G = T(:, :, 2, :);
e = 1e-6;
Pc = min(max(P, e), 1 - e);
logP = log(Pc);
logN = log(1 - Pc);
nV = sum(G, 'all'); nB = sum(1 - G, 'all');
wV = min(max(nB / max(nV, 1), 1), 100);
cePart = -mean(G .* (wV/W0) .* logP + (1 - G) .* logN, 'all');
smooth = 1;
inter = sum(P .* G, 'all');
dice = (2*inter + smooth) / (sum(P, 'all') + sum(G, 'all') + smooth);
dicePart = 1 - dice;
loss = cePart + dicePart;
gradients = dlgradient(loss, net.Learnables);
end
