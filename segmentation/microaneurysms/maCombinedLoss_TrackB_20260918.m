function loss = maCombinedLoss_TrackB_20260918(Y, T)
% maCombinedLoss_TrackB_20260918 — Phase 6 model loss. EXPERIMENTAL.
% This module is experimental. Never describe it as clinically validated.
%
% R2026a trainnet convention: lossFcn(Y, T).
%   Y = network predictions (unet already ends in softmax), HxWxCxN dlarray.
%   T = one-hot targets HxWxCxN (ch1 = background, ch2 = MA).
% Combined moderated weighted CE [0.5 1.5] (prior-session fix for the
% inverse-frequency overprediction) + soft Dice on the MA channel.

ce = crossentropy(Y, T, [0.5 1.5], ...
    WeightsFormat = "UC", NormalizationFactor = "all-elements");

pMA = Y(:, :, 2, :);
tMA = T(:, :, 2, :);
smooth = 1e-6;
inter = sum(pMA .* tMA, [1 2]);
denom = sum(pMA, [1 2]) + sum(tMA, [1 2]) + smooth;
diceLoss = mean(1 - (2 * inter + smooth) ./ denom);

loss = ce + diceLoss;
end
