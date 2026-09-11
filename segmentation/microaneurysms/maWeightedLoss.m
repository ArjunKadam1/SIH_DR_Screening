function loss = maWeightedLoss(Y,T)
% maWeightedLoss
% Class-weighted cross-entropy loss for microaneurysm segmentation.
%
% Class order:
%   1 = background
%   2 = microaneurysm

classWeights = [0.0051 1.9949];

loss = crossentropy( ...
    Y, ...
    T, ...
    classWeights, ...
    WeightsFormat="UC", ...
    NormalizationFactor="all-elements");

end