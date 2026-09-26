function T = calibrateTemperature(net, augVal, valLabels, varargin)
% CALIBRATETEMPERATURE - single-parameter confidence calibration (Guo et al.)
%
% Fits temperature T > 0 on VALIDATION data only (never test) so reported
% confidence matches empirical accuracy. Fixes the common "99% confident
% but wrong" overconfidence of softmax classifiers without retraining.
%
% Usage:
%   T = calibrateTemperature(net, augVal, valLabels)  % saves results/temperature_v1.mat
%   T = calibrateTemperature(net, augVal, valLabels, 'SavePath', p)
%
% Inputs:
%   net       - trained DAGNetwork (V1 or V2)
%   augVal    - augmentedImageDatastore for val set (no augmentation)
%   valLabels - categorical ground truth for val set
%
% Output:
%   T - scalar temperature (T=1 = uncalibrated; T>1 softens overconfidence)
%
% Applies as: calibrated_scores = softmax(log(raw_scores) / T).
% Demo uses it automatically if results/temperature_v1.mat exists.
p = inputParser;
addParameter(p,'SavePath',"",@(x) isstring(x) || ischar(x));
parse(p,varargin{:});
projectRoot = fileparts(fileparts(mfilename('fullpath')));
savePath = string(p.Results.SavePath);
if savePath == ""
    savePath = fullfile(projectRoot,'results','temperature_v1.mat');
end
[~, scores] = classify(net, augVal);
scores = double(scores);
scores = max(scores, 1e-12);
logits = log(scores);
Y = valLabels;
% Grid search T in [0.5, 5] minimizing NLL on val
Ts = 0.5:0.05:5;
bestNLL = inf; T = 1;
for t = Ts
    cal = softmax2(logits / t);
    nll = -mean(log(max(cal(sub2ind(size(cal), (1:numel(Y))', double(Y))), 1e-12)));
    if nll < bestNLL
        bestNLL = nll; T = t;
    end
end
% Report ECE before/after (15 bins)
[eCEBefore, accB, confB] = ece(scores, Y);
[eCEAfter, accA, confA] = ece(softmax2(logits / T), Y);
fprintf('calibrateTemperature: T=%.2f NLL=%.4f ECE %.3f->%.3f (acc %.1f%%, conf %.1f%%->%.1f%%)\n', ...
    T, bestNLL, eCEBefore, eCEAfter, accB*100, confB*100, confA*100);
save(savePath, 'T');
fprintf('Saved %s\n', savePath);
end

function S = softmax2(X)
E = exp(X - max(X,[],2));
S = E ./ sum(E,2);
end

function [e, acc, conf] = ece(scores, Y)
[~, pred] = max(scores,[],2);
classes = categories(Y);
predCat = categorical(classes(pred), categories(Y));
correct = predCat == Y;
confAll = max(scores,[],2);
acc = mean(correct); conf = mean(confAll);
e = 0; edges = linspace(0,1,16);
for b = 1:15
    idx = confAll > edges(b) & confAll <= edges(b+1);
    if any(idx)
        e = e + sum(idx)/numel(Y) * abs(mean(correct(idx)) - mean(confAll(idx)));
    end
end
end
