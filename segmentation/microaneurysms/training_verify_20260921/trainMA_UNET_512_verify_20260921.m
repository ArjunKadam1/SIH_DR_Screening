% trainMA_UNET_512_verify_20260921
% NEW implementation (2026-09-21). Real 1-epoch MA U-Net training + eval.
% Design (approved): plain SGD lr 1e-4 (no Adam), strict 512 crops
% (no padding), rng(42) 50-patch eval protocol.
% Data: Downloads/MA_Patch_Manifests (train 4948 / val 1258), IDRiD+eOphtha.
% Net: friend's MA_UNET_baseline_architecture (loaded, never modified on disk).
% Loss/grads: friend's MA_modelGradients (Dice+Focal) via dlfeval.
% Writes ONLY to results/MA_verify_20260921/. Never overwrites: asserts
% output files do not exist. STOP on non-finite loss.
% Flow: 1 epoch (1237 iters, batch 4) -> full val -> 50-patch Dice/IoU ->
% 3 overlays -> save + report -> STOP (no 15-epoch).

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
manDir   = 'C:/Users/SHIVANYA SALES/Downloads/MA_Patch_Manifests';
outDir   = fullfile(projRoot, 'results/MA_verify_20260921');
addpath('C:/Users/SHIVANYA SALES/Downloads'); % MA_modelGradients.m
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_verify_20260921')); % cropMA512.m

outNet = fullfile(outDir, 'MA_UNET_512_1epoch_VERIFY_20260921.mat');
outLog = fullfile(outDir, 'train_1epoch_20260921.csv');
assert(~isfile(outNet), 'Output exists — STOP (never overwrite).');

seed = 42; rng(seed);
lr = 1e-4; bs = 4;
useGPU = (gpuDeviceCount > 0);
fprintf('SEED=%d LR=%.0e BATCH=%d GPU=%d\n', seed, lr, bs, useGPU);

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
    [loss, grads] = dlfeval(@MA_modelGradients, net, Xd, Td);
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

% ---- full validation (1258 patches) ----
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
    P = (1./(1+exp(-extractdata(gather(Y(:, :, 2, :))))));
    G = Tb(:, :, 2, :);
    smooth = 1;
    inter = sum(P .* G, [1 2]);
    dice = (2*inter + smooth) ./ (sum(P, [1 2]) + sum(G, [1 2]) + smooth);
    valLoss(it) = gather(1 - mean(dice));
    if mod(it, 50) == 0 || it == nV, fprintf('VAL iter=%d/%d\n', it, nV); end
end
fprintf('VAL_DONE meanDiceLoss=%.4f\n', mean(valLoss));

% ---- 50-patch eval, rng(42) protocol ----
rng(42);
sel = sort(randperm(height(v), 50));
diceS = zeros(50, 1); iouS = zeros(50, 1);
for k = 1:50
    [Xb, Tb] = cropMA512(v, sel(k));
    if useGPU, Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
    else,      Xd = dlarray(single(Xb), 'SSCB'); end
    Y = forward(net, Xd);
    P = (1./(1+exp(-extractdata(gather(Y(:, :, 2, :)))))) >= 0.5;
    G = logical(Tb(:, :, 2, :));
    inter = nnz(P & G);
    diceS(k) = 2*inter / (nnz(P) + nnz(G) + eps);
    iouS(k)  = inter / (nnz(P | G) + eps);
end
fprintf('EVAL50 Dice mean=%.4f med=%.4f | IoU mean=%.4f med=%.4f\n', ...
    mean(diceS), median(diceS), mean(iouS), median(iouS));

% ---- 3 overlays: best / median / worst by Dice ----
[~, ordD] = sort(diceS);
picks = [ordD(end), ordD(25), ordD(1)];
tags = ["best", "median", "worst"];
for k = 1:3
    [Xb, ~] = cropMA512(v, sel(picks(k)));
    if useGPU, Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
    else,      Xd = dlarray(single(Xb), 'SSCB'); end
    Y = forward(net, Xd);
    P = (1./(1+exp(-extractdata(gather(Y(:, :, 2, :)))))) >= 0.5;
    Ip = uint8(255 * Xb(:, :, :, 1));
    ov = Ip;
    Pm = repmat(P, [1 1 3]);
    red = cat(3, ones(512, 512, 'uint8')*255, zeros(512, 512, 'uint8'), zeros(512, 512, 'uint8'));
    ov(Pm) = uint8(double(Ip(Pm))*0.4 + double(red(Pm))*0.6);
    imwrite(ov, fullfile(outDir, sprintf('overlay_%s_dice%.3f.png', tags(k), diceS(picks(k)))));
end

params = struct('seed', seed, 'lr', lr, 'batch', bs, 'optimizer', 'SGD', ...
    'patchSize', 512, 'useGPU', useGPU, 'nIter', nIter);
eval50 = struct('sel', sel, 'dice', diceS, 'iou', iouS);
save(outNet, 'net', 'trainLoss', 'valLoss', 'params', 'eval50', 'order');
fprintf('SAVED=%s\n', outNet);
fprintf('ALL_DONE STOP (no 15-epoch training)\n');
