function R = analyzeNet_ACTFIX_20260921(net, tag)
% analyzeNet_ACTFIX_20260921 — NEW (branch MA_ACTFIX_20260921).
% Sigmoid-free analysis: P = raw softmax channel 2 (already in [0,1]).
% Same rng(42) 50 val patches as the failed run. Returns struct R and
% saves rank_<tag>_ACTFIX_20260921.mat/.csv into results/MA_ACTFIX_20260921/.
projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
outDir   = fullfile(projRoot, 'results/MA_ACTFIX_20260921');
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_verify_20260921')); % cropMA512.m
V = load('C:/Users/SHIVANYA SALES/Downloads/MA_Patch_Manifests/MA_val_manifest.mat');
v = V.valManifest;
useGPU = (gpuDeviceCount > 0);

rng(42);
sel = sort(randperm(height(v), 50));
isPos = false(50, 1);
meanP_MA = nan(50, 1); meanP_BG = zeros(50, 1); fpFrac = zeros(50, 1);
dice05 = nan(50, 1);
ths = 0.05:0.05:0.9;
diceSweep = nan(50, numel(ths));
aucPatch = nan(50, 1); apPatch = nan(50, 1);
 Pall = []; Gall = [];
for k = 1:50
    [Xb, Tb] = cropMA512(v, sel(k));
    if useGPU, Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
    else,      Xd = dlarray(single(Xb), 'SSCB'); end
    Y = forward(net, Xd);
    P = double(extractdata(gather(Y(:, :, 2, :)))); % NO sigmoid: softmax already
    G = logical(Tb(:, :, 2, :));
    isPos(k) = any(G(:));
    meanP_BG(k) = mean(P(~G));
    if isPos(k)
        meanP_MA(k) = mean(P(G));
        Pt = P >= 0.5;
        dice05(k) = 2*nnz(Pt & G) / (nnz(Pt) + nnz(G) + eps);
        for t = 1:numel(ths)
            Q = P >= ths(t);
            diceSweep(k, t) = 2*nnz(Q & G) / (nnz(Q) + nnz(G) + eps);
        end
        [aucPatch(k), apPatch(k)] = rocAP(P(:), double(G(:)));
        Pall = [Pall; P(:)]; Gall = [Gall; double(G(:))]; %#ok<AGROW>
    else
        fpFrac(k) = mean(P(:) >= 0.5);
    end
    if mod(k, 10) == 0, fprintf('[%s] ANALYZED %d/50\n', tag, k); end
end
[aucPooled, apPooled] = rocAP(Pall, Gall);
[bestDice, bi] = max(mean(diceSweep, 1, 'omitnan'));

R = struct('tag', tag, 'sel', sel, 'isPos', isPos, ...
    'meanP_MA', meanP_MA, 'meanP_BG', meanP_BG, 'fpFracNeg', mean(fpFrac(~isPos)), ...
    'dice05meanPos', mean(dice05, 'omitnan'), 'dice05medPos', median(dice05, 'omitnan'), ...
    'gapMeanPos', mean(meanP_MA(isPos) - meanP_BG(isPos), 'omitnan'), ...
    'aucPooled', aucPooled, 'apPooled', apPooled, ...
    'aucMedPatch', median(aucPatch, 'omitnan'), 'apMedPatch', median(apPatch, 'omitnan'), ...
    'bestDice', bestDice, 'bestThr', ths(bi));
fprintf('[%s] nPos=%d FPfracNeg=%.5f Dice05mean=%.4f gap=%.5f AUCpool=%.4f APpool=%.4f bestDice=%.4f@thr=%.2f\n', ...
    tag, nnz(isPos), R.fpFracNeg, R.dice05meanPos, R.gapMeanPos, ...
    R.aucPooled, R.apPooled, R.bestDice, R.bestThr);

M = table(sel(:), isPos, meanP_MA, meanP_BG, dice05, aucPatch, apPatch, 'VariableNames', ...
    {'row', 'isPos', 'meanP_MA', 'meanP_BG', 'dice05', 'auroc', 'ap'});
writetable(M, fullfile(outDir, sprintf('rank_%s_ACTFIX_20260921.csv', tag)));
save(fullfile(outDir, sprintf('rank_%s_ACTFIX_20260921.mat', tag)), 'R');
end

function [auroc, ap] = rocAP(scores, labels)
% Threshold-free ranking metrics, no toolbox needed.
[ss, ord] = sort(scores, 'descend');
lab = labels(ord);
P = sum(lab); N = numel(lab) - P;
if P == 0 || N == 0, auroc = nan; ap = nan; return; end
tp = cumsum(lab); fp = cumsum(~lab);
tpr = tp / P; fpr = fp / N;
te = [0; tpr]; fe = [0; fpr];
auroc = sum(diff(fe) .* (te(1:end-1) + te(2:end)) / 2);
prec = tp ./ max(tp + fp, 1);
rec = tpr;
ap = sum(diff([0; rec]) .* prec);
end
