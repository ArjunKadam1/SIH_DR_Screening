% trainMA_UNET_512_ACTFIX_20260921
% NEW (branch MA_ACTFIX_20260921). 1-epoch SGD, lr 1e-4, batch 4, seed 42.
% Same init (baseline .mat) + same data order as the failed verify run:
% the ONLY difference is the fixed loss (no sigmoid).
% Per-epoch checkpoint + per-50-iter CSV. Full val, trained analysis,
% 3 overlays, gate verdict vs locked G1/G2. Then STOP.

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
manDir   = 'C:/Users/SHIVANYA SALES/Downloads/MA_Patch_Manifests';
outDir   = fullfile(projRoot, 'results/MA_ACTFIX_20260921');
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_ACTFIX_20260921'));
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_verify_20260921'));

outNet = fullfile(outDir, 'MA_UNET_512_1epoch_ACTFIX_20260921.mat');
outLog = fullfile(outDir, 'train_1epoch_ACTFIX_20260921.csv');
assert(~isfile(outNet), 'Output exists — STOP (never overwrite).');

seed = 42; rng(seed);
lr = 1e-4; bs = 4;
useGPU = (gpuDeviceCount > 0);
fprintf('ACTFIX SEED=%d LR=%.0e BATCH=%d GPU=%d\n', seed, lr, bs, useGPU);

T = load(fullfile(manDir, 'MA_train_manifest.mat')); m = T.manifest;
V = load(fullfile(manDir, 'MA_val_manifest.mat'));   v = V.valManifest;
B = load('C:/Users/SHIVANYA SALES/Downloads/MA_UNET_baseline_architecture.mat');
net = B.net;

order = randperm(height(m));
nIter = ceil(height(m) / bs);
trainLoss = zeros(nIter, 1);
t0 = tic;
logF = fopen(outLog, 'w'); fprintf(logF, 'iter,loss\n');
for it = 1:nIter
    idx = order((it-1)*bs+1:min(it*bs, height(m)));
    [Xb, Tb] = cropMA512(m, idx);
    if useGPU
        Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
        Td = gpuArray(dlarray(single(Tb), 'SSCB'));
    else
        Xd = dlarray(single(Xb), 'SSCB');
        Td = dlarray(single(Tb), 'SSCB');
    end
    [loss, grads] = dlfeval(@MA_modelGradients_ACTFIX_20260921, net, Xd, Td);
    lv = gather(extractdata(loss));
    assert(isfinite(lv), sprintf('Non-finite train loss at iter %d — STOP.', it));
    trainLoss(it) = lv;
    net.Learnables = dlupdate(@(p, g) p - lr*g, net.Learnables, grads);
    fprintf(logF, '%d,%.6f\n', it, lv);
    if mod(it, 50) == 0 || it == nIter
        fprintf('TRAIN iter=%d/%d loss=%.4f elapsed=%.1fmin\n', it, nIter, lv, toc(t0)/60);
    end
end
fclose(logF);
fprintf('EPOCH_DONE iters=%d meanLoss=%.4f elapsedMin=%.1f\n', nIter, mean(trainLoss), toc(t0)/60);

params = struct('seed', seed, 'lr', lr, 'batch', bs, 'optimizer', 'SGD', ...
    'patchSize', 512, 'useGPU', useGPU, 'nIter', nIter, 'branch', 'MA_ACTFIX_20260921');
save(outNet, 'net', 'trainLoss', 'params', 'order'); % per-epoch checkpoint
fprintf('CHECKPOINT_SAVED=%s\n', outNet);

% ---- full validation (fixed-loss Dice) ----
nV = ceil(height(v) / bs);
valLoss = zeros(nV, 1);
for it = 1:nV
    idx = (it-1)*bs+1:min(it*bs, height(v));
    [Xb, Tb] = cropMA512(v, idx);
    if useGPU
        Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
        Td = gpuArray(dlarray(single(Tb), 'SSCB'));
    else
        Xd = dlarray(single(Xb), 'SSCB');
        Td = dlarray(single(Tb), 'SSCB');
    end
    Y = forward(net, Xd);
    P = extractdata(gather(Y(:, :, 2, :))); % NO sigmoid (softmax already)
    G = Tb(:, :, 2, :);
    smooth = 1;
    dice = (2*sum(P .* G, [1 2]) + smooth) ./ (sum(P, [1 2]) + sum(G, [1 2]) + smooth);
    valLoss(it) = gather(1 - mean(dice));
end
fprintf('VAL_DONE meanDiceLoss=%.4f\n', mean(valLoss));

% ---- trained analysis (same 50 patches, sigmoid-free) ----
Rtr = analyzeNet_ACTFIX_20260921(net, 'trained');

% ---- 3 overlays: best / median / worst Dice@0.5 among positives ----
sel = Rtr.sel;
D = nan(50, 1);
for k = 1:50
    [~, Tb] = cropMA512(v, sel(k));
    if any(Tb(:, :, 2, :) > 0, 'all')
        [Xb, ~] = cropMA512(v, sel(k));
        if useGPU, Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
        else,      Xd = dlarray(single(Xb), 'SSCB'); end
        Y = forward(net, Xd);
        P = double(extractdata(gather(Y(:, :, 2, :)))) >= 0.5;
        G = logical(Tb(:, :, 2, :));
        D(k) = 2*nnz(P & G) / (nnz(P) + nnz(G) + eps);
    end
end
Dv = D(~isnan(D)); [~, so] = sort(Dv);
posIdx = find(~isnan(D));
picks = [posIdx(so(end)), posIdx(so(ceil(numel(so)/2))), posIdx(so(1))];
tags = ["best", "median", "worst"];
for k = 1:3
    [Xb, ~] = cropMA512(v, sel(picks(k)));
    if useGPU, Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
    else,      Xd = dlarray(single(Xb), 'SSCB'); end
    Y = forward(net, Xd);
    P = double(extractdata(gather(Y(:, :, 2, :)))) >= 0.5;
    Ip = uint8(255 * Xb(:, :, :, 1));
    Pm = repmat(P, [1 1 3]);
    red = cat(3, ones(512, 512, 'uint8')*255, zeros(512, 512, 'uint8'), zeros(512, 512, 'uint8'));
    ov = Ip; ov(Pm) = uint8(double(Ip(Pm))*0.4 + double(red(Pm))*0.6);
    imwrite(ov, fullfile(outDir, sprintf('overlay_%s_dice%.3f.png', tags(k), D(picks(k)))));
end

% ---- gate verdict (locked) ----
G1 = Rtr.dice05meanPos > 0.05;
G2 = Rtr.gapMeanPos >= 0.02;
fprintf('GATE G1(Dice05>0.05)=%d value=%.4f | G2(gap>=0.02)=%d value=%.5f\n', ...
    G1, Rtr.dice05meanPos, G2, Rtr.gapMeanPos);
fprintf('REPORT-ONLY AUCpool=%.4f APpool=%.4f bestDice=%.4f@%.2f\n', ...
    Rtr.aucPooled, Rtr.apPooled, Rtr.bestDice, Rtr.bestThr);
save(outNet, 'net', 'trainLoss', 'valLoss', 'params', 'order', 'Rtr', 'G1', 'G2');
fprintf('ALL_DONE STOP\n');
