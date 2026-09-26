function [loss, gradients, cePart, dicePart, wV] = vesselLoss_dice5(net, X, T, W0)
% vesselLoss_dice5 — P4 loss (new file; frozen + norm untouched).
% Single change vs vesselLoss_norm: Dice term weighted x5.
% loss = cePart + 5*dicePart, with cePart vessel-term normalized by FIXED W0.
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
loss = cePart + 5*dicePart;
gradients = dlgradient(loss, net.Learnables);
end
