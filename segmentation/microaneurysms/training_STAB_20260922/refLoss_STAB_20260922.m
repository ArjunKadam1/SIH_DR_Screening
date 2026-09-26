function [loss, gradients, cePart, dicePart] = refLoss_STAB_20260922(net, X, T)
% refLoss_STAB_20260922 — NEW (branch MA_STAB_20260922, Phase 1).
% EXACT copy of D3 refLoss_ACTFIX math (verified identical term-by-term):
% inverse-frequency weighted CE from log-probs (clamp [1e-6,1-1e-6]) +
% batch soft-Dice, smooth=1. Only the clipping (in trainer) is new.
% Returns CE and Dice parts separately for per-checkpoint logging.
Y = forward(net, X);
P = Y(:, :, 2, :);
G = T(:, :, 2, :);
e = 1e-6;
Pc = min(max(P, e), 1 - e);
logP = log(Pc);
logN = log(1 - Pc);
nMA = sum(G, 'all'); nBG = sum(1 - G, 'all');
wMA = min(max(nBG / max(nMA, 1), 1), 100);
cePart = -mean(G .* wMA .* logP + (1 - G) .* logN, 'all');
smooth = 1;
inter = sum(P .* G, 'all');
dice = (2*inter + smooth) / (sum(P, 'all') + sum(G, 'all') + smooth);
dicePart = 1 - dice;
loss = cePart + dicePart;
gradients = dlgradient(loss, net.Learnables);
end
