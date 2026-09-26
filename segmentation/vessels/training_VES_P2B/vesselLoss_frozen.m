function [loss, gradients, cePart, dicePart] = vesselLoss_frozen(net, X, T)
% vesselLoss_frozen — FROZEN loss for VES P2B both arms (Arm A resize + Arm B native).
% Math: inverse-frequency weighted CE from log-probs (clamp [1e-6,1-1e-6]) +
% batch soft-Dice, smooth=1. Mirrors MA STAB refLoss term-by-term.
% X: dlarray SSCB 256x256x3xN single. T: dlarray SSCB 256x256x2xN single (ch1=bg,ch2=vessel).
% NOTE: original 5-epoch vessel run used trainNetwork with LR 1e-3 and unknown
% loss details (no vessel loss code exists in repo). This frozen loss is the
% documented loss for P2B — identical across arms, so the resize-vs-native
% comparison stays single-variable even though the loss itself is newly specified.
Y = forward(net, X);
P = Y(:, :, 2, :);
G = T(:, :, 2, :);
e = 1e-6;
Pc = min(max(P, e), 1 - e);
logP = log(Pc);
logN = log(1 - Pc);
nV = sum(G, 'all'); nB = sum(1 - G, 'all');
wV = min(max(nB / max(nV, 1), 1), 100);
cePart = -mean(G .* wV .* logP + (1 - G) .* logN, 'all');
smooth = 1;
inter = sum(P .* G, 'all');
dice = (2*inter + smooth) / (sum(P, 'all') + sum(G, 'all') + smooth);
dicePart = 1 - dice;
loss = cePart + dicePart;
gradients = dlgradient(loss, net.Learnables);
end
