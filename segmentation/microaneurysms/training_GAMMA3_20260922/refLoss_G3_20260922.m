function [loss, gradients] = refLoss_G3_20260922(net, X, T)
% refLoss_G3_20260922 — NEW (branch MA_GAMMA3_20260922).
% SOLE CHANGE vs ACTFIX loss (MA_modelGradients_ACTFIX_20260921):
% focal gamma 2 -> 3. Everything else identical: Dice smooth=1 (per-batch),
% eps=1e-6 in both logs, NO alpha, P=Y(:,:,2,:) from softmax net.
% Rationale (locked): test whether a sharper easy/hard split fixes the
% scattered-FP collapse (FPLOC) on the exact rig where gamma=2 failed.
Y = forward(net, X);
P = Y(:, :, 2, :);
G = T(:, :, 2, :);
smooth = 1;
intersection = sum(P.*G, 'all');
dice = (2*intersection + smooth) ./ ...
    (sum(P, 'all') + sum(G, 'all') + smooth);
diceLoss = 1 - dice;
epsVal = 1e-6;
focal = -mean(G.*(1-P).^3.*log(P+epsVal) + ...
    (1-G).*P.^3.*log(1-P+epsVal), 'all');
loss = diceLoss + focal;
gradients = dlgradient(loss, net.Learnables);
end
