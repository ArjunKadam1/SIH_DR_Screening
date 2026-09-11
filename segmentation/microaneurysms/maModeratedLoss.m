function loss = maModeratedLoss(Y,T)
% maModeratedLoss
% Moderated weighted cross-entropy for microaneurysm segmentation.
%
% Background = 0.5
% Microaneurysm = 1.5
%
% The previous inverse-frequency weighting caused severe
% microaneurysm overprediction.

classWeights = [0.5 1.5];

loss = crossentropy( ...
    Y, ...
    T, ...
    classWeights, ...
    WeightsFormat="UC", ...
    NormalizationFactor="all-elements");

end