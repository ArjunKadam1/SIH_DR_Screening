function T = calibrateTemperature_FIXED_20260918(net, augVal, valLabels, varargin)
% CALIBRATETEMPERATURE_FIXED_20260918 - dated copy of calibrateTemperature.m
% with the class-order alignment fix (original untouched).
%
% Bug fixed: the original indexes score columns with double(valLabels),
% whose indices follow the CALLER's categorical order, while classify()
% columns follow NET class order (alphabetical: Mild,Moderate,No_DR,
% SevereProlif). Any caller passing non-net-ordered labels silently fits T
% against wrong-class probabilities. This copy re-orders valLabels to net
% order internally, so the fix holds regardless of caller convention.
% Grid (0.5:0.05:5), NLL objective, 15-bin ECE, SavePath: all unchanged.
p = inputParser;
addParameter(p,'SavePath',"",@(x) isstring(x) || ischar(x));
parse(p,varargin{:});
projectRoot = 'C:\Users\SHIVANYA SALES\Desktop\DR tejas\SIH_DR_Screening';
savePath = string(p.Results.SavePath);
if savePath == ""
    savePath = fullfile(projectRoot,'results','temperature_v1.mat');
end
% FIX (20260918): single source of truth = trainer ICDR order.
% The net stores classes alphabetically; every index-based use below runs
% in TRAINER order after an explicit column permutation. Never trust net
% order implicitly.
trainerOrder = {'No_DR','Mild','Moderate','SevereProlif'};
netOrder = cellstr(string(net.Layers(end).ClassNames));
if ~isequal(sort(netOrder(:)), sort(trainerOrder(:)))
    error('calibrateTemperature_FIXED: net classes != trainer classes.');
end
[~, perm] = ismember(trainerOrder, netOrder);  % trainer idx -> net col
Y = categorical(string(valLabels), trainerOrder);
[~, scores] = classify(net, augVal);
scores = double(scores);
scores = scores(:, perm);  % remap net-alpha columns -> trainer ICDR order
scores = max(scores, 1e-12);
logits = log(scores);
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
fprintf('calibrateTemperature_FIXED: T=%.2f NLL=%.4f ECE %.3f->%.3f (acc %.1f%%, conf %.1f%%->%.1f%%)\n', ...
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
