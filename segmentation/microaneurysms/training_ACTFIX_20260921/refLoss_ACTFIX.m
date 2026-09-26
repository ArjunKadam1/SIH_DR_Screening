function [loss, gradients] = refLoss_ACTFIX(net, X, T)
% refLoss_ACTFIX — NEW (branch MA_ACTFIX_20260921, D3 reference loss).
% Inverse-frequency weighted CE from LOG-probabilities (clamped) + batch
% soft-Dice, smooth=1. Class weights: wBG=1, wMA=nBG/nMA clamped [1,100].
Y = forward(net, X);
P = Y(:, :, 2, :);
G = T(:, :, 2, :);
e = 1e-6;
Pc = min(max(P, e), 1 - e);
logP = log(Pc);
logN = log(1 - Pc);
nMA = sum(G, 'all'); nBG = sum(1 - G, 'all');
wMA = min(max(nBG / max(nMA, 1), 1), 100);
ce = -mean(G .* wMA .* logP + (1 - G) .* logN, 'all');
smooth = 1;
inter = sum(P .* G, 'all');
dice = (2*inter + smooth) / (sum(P, 'all') + sum(G, 'all') + smooth);
loss = ce + (1 - dice);
gradients = dlgradient(loss, net.Learnables);
end
