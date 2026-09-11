function loss = maDiceLoss(Y,T)
% maDiceLoss
%
% Soft Dice loss for microaneurysm segmentation.
%
% Y = predicted softmax probabilities
% T = one-hot encoded ground truth
%
% Dimensions:
%   H x W x Classes x Batch
%
% Channel 1 = background
% Channel 2 = microaneurysm

    % ---------------------------------------------------------
    % Extract microaneurysm channel
    % ---------------------------------------------------------

    pMA = Y(:,:,2,:);

    % Ground-truth MA channel
    tMA = T(:,:,2,:);

    % ---------------------------------------------------------
    % Soft Dice
    % ---------------------------------------------------------

    smooth = 1e-6;

    intersection = sum(pMA .* tMA,'all');

    predictionSum = sum(pMA,'all');

    targetSum = sum(tMA,'all');

    diceScore = ...
        (2*intersection + smooth) ./ ...
        (predictionSum + targetSum + smooth);

    % ---------------------------------------------------------
    % Dice loss
    % ---------------------------------------------------------

    loss = 1 - diceScore;

end