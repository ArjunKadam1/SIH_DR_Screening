function [loss, gradients] = MA_modelGradients_ACTFIX_20260921(net, X, T)
% MA_modelGradients_ACTFIX_20260921 — NEW (2026-09-21, branch MA_ACTFIX_20260921).
% One-variable fix vs friend's MA_modelGradients.m: the U-Net ends in a
% SoftmaxLayer, so Y(:,:,2,:) is ALREADY a probability in [0,1].
% Friend's version applied sigmoid() on top, crushing P into [0.50,0.73].
% Fix: P = Y(:,:,2,:) directly. Everything else identical (Dice smooth=1,
% focal gamma=2, eps=1e-6, dlgradient w.r.t. Learnables).
Y = forward(net, X);
P = Y(:, :, 2, :);
G = T(:, :, 2, :);
smooth = 1;
intersection = sum(P.*G, 'all');
dice = (2*intersection + smooth) ./ ...
    (sum(P, 'all') + sum(G, 'all') + smooth);
diceLoss = 1 - dice;
epsVal = 1e-6;
focal = -mean(G.*(1-P).^2.*log(P+epsVal) + ...
    (1-G).*P.^2.*log(1-P+epsVal), 'all');
loss = diceLoss + focal;
gradients = dlgradient(loss, net.Learnables);
end
