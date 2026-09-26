function [loss, gradients, focalPart, dicePart] = vesselLoss_focal(net, X, T)
% vesselLoss_focal — P5 loss (new file; frozen/norm/dice5 untouched).
% Single change vs vesselLoss_dice5: focal vessel term (gamma=2, NO alpha,
% eps via vessel-line [1e-6,1-1e-6] clamp) takes CE's slot; batch Dice x5 kept.
% Pattern mirrors MA_modelGradients (focal gamma=2, NO alpha) except Dice x5
% (deliberate: single-variable vs P4; precedent uses x1) and clamp-vs-additive
% eps (deliberate: vessel-line consistency). No wV/W0 (focal needs none).
Y = forward(net, X);
P = Y(:, :, 2, :);
G = T(:, :, 2, :);
e = 1e-6;
Pc = min(max(P, e), 1 - e);
focalPart = -mean(G .* (1-Pc).^2 .* log(Pc) + (1-G) .* Pc.^2 .* log(1-Pc), 'all');
smooth = 1;
inter = sum(P .* G, 'all');
dice = (2*inter + smooth) / (sum(P, 'all') + sum(G, 'all') + smooth);
dicePart = 1 - dice;
loss = focalPart + 5*dicePart;
gradients = dlgradient(loss, net.Learnables);
end
