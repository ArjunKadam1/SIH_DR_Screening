% analyze_rank_20260921
% ANALYSIS ONLY (2026-09-21). No training, no weight updates.
% Loads MA_UNET_512_1epoch_VERIFY_20260921.mat and asks: does the net rank
% MA pixels above background pixels at all?
% Metrics per patch (same rng(42) 50-patch set as eval, stored in .mat):
%  - mean P on MA pixels vs mean P on background pixels (separation)
%  - Dice at thresholds [0.05 0.1 0.2 0.3 0.5 0.7]
% Writes only results/MA_verify_20260921/rank_analysis_20260921.mat + .csv.

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
outDir   = fullfile(projRoot, 'results/MA_verify_20260921');
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_verify_20260921'));
addpath('C:/Users/SHIVANYA SALES/Downloads');

S = load(fullfile(outDir, 'MA_UNET_512_1epoch_VERIFY_20260921.mat'), 'net', 'eval50');
net = S.net; sel = S.eval50.sel;
V = load('C:/Users/SHIVANYA SALES/Downloads/MA_Patch_Manifests/MA_val_manifest.mat');
v = V.valManifest;
useGPU = (gpuDeviceCount > 0);
ths = [0.05 0.1 0.2 0.3 0.5 0.7];

meanP_MA = zeros(50, 1); meanP_BG = zeros(50, 1);
diceT = zeros(50, numel(ths));
for k = 1:50
    [Xb, Tb] = cropMA512(v, sel(k));
    if useGPU, Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
    else,      Xd = dlarray(single(Xb), 'SSCB'); end
    Y = forward(net, Xd);
    P = extractdata(gather(Y(:, :, 2, :)));
    P = 1 ./ (1 + exp(-P));
    G = logical(Tb(:, :, 2, :));
    meanP_MA(k) = mean(P(G));
    meanP_BG(k) = mean(P(~G));
    for t = 1:numel(ths)
        Pt = P >= ths(t);
        inter = nnz(Pt & G);
        diceT(k, t) = 2*inter / (nnz(Pt) + nnz(G) + eps);
    end
    if mod(k, 10) == 0, fprintf('ANALYZED %d/50\n', k); end
end

fprintf('== SEPARATION ==\n');
fprintf('meanP(MA) overall=%.5f | meanP(BG) overall=%.5f | ratio=%.2f\n', ...
    mean(meanP_MA), mean(meanP_BG), mean(meanP_MA)/max(mean(meanP_BG), eps));
fprintf('patches with meanP(MA)>meanP(BG): %d/50\n', nnz(meanP_MA > meanP_BG));
fprintf('== DICE BY THRESHOLD ==\n');
for t = 1:numel(ths)
    fprintf('thr=%.2f meanDice=%.4f medDice=%.4f\n', ths(t), mean(diceT(:, t)), median(diceT(:, t)));
end

T = table(sel(:), meanP_MA, meanP_BG, diceT(:,1), diceT(:,2), diceT(:,3), ...
    diceT(:,4), diceT(:,5), diceT(:,6), 'VariableNames', ...
    {'row', 'meanP_MA', 'meanP_BG', 'd_005', 'd_01', 'd_02', 'd_03', 'd_05', 'd_07'});
writetable(T, fullfile(outDir, 'rank_analysis_20260921.csv'));
save(fullfile(outDir, 'rank_analysis_20260921.mat'), 'sel', 'meanP_MA', 'meanP_BG', 'diceT', 'ths');
fprintf('RANK_ANALYSIS_DONE\n');
