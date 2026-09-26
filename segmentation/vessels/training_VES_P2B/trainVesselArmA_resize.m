% trainVesselArmA_resize — VES P2B Arm A (control: resize, matched training).
% Resumes vesselNetWeightedTrained byte-identical, old resize-to-256 pipeline
% unchanged, 16 train / 4 val image-level split, 10 epochs, per-epoch ckpts.
% Frozen: Adam 1e-4, batch 4, clip-norm 1.0, rng(42), vesselLoss_frozen,
% same DRIVE split, aug none. Gates: NaN/Inf STOP, norm>1e6 STOP,
% loss>5 sustained 50 iters STOP. Test-20 untouched.
% NOTE: original 5-epoch run used LR 1e-3; P2B uses 1e-4 per frozen spec —
% intentional optimizer change, identical across arms (single-variable holds
% for resize-vs-native, not vs-original).

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
addpath(fullfile(projRoot, 'segmentation/vessels/training_VES_P2B'));
addpath(fullfile(projRoot, 'segmentation/vessels'));
stamp = datestr(now, 'yyyymmdd_HHMMSS');
outDir = fullfile(projRoot, 'results', ['VES_P2B_256resize_' stamp]);
mkdir(outDir);
logPath = fullfile(outDir, 'console_ArmA.log');
diary(logPath);
fprintf('ARM-A RESIZE-CONTROL stride=N/A(resize) rng=42 lr=1e-4 bs=4 clip=1.0 epochs=10\n');

% --- fixed image-level split: train 21-36 (16), val 37-40 (4) ---
trainIds = 21:36; valIds = 37:40;
fprintf('SPLIT train=%s val=%s (image-level, test 01-20 blind)\n', ...
    mat2str(trainIds), mat2str(valIds));

% --- byte-identical resume check ---
srcFile = fullfile(projRoot, 'results/vesselNet_weighted_5epoch.mat');
S = load(srcFile, 'vesselNetWeightedTrained');
net = S.vesselNetWeightedTrained;
% fingerprint: total learnable norm + first/last conv weight sums
vals = net.Learnables.Value;
totN = 0; cks = zeros(numel(vals), 1);
for i = 1:numel(vals)
    v = double(gather(extractdata(vals{i})));
    totN = totN + sum(v(:).^2);
    cks(i) = sum(v(:));
end
fpNorm = sqrt(totN);
fprintf('RESUME-CHECK var=vesselNetWeightedTrained layers=%d fpNorm=%.6f ckSum=%.6f nTensors=%d\n', ...
    numel(net.Layers), fpNorm, sum(cks), numel(vals));
fprintf('RESUME-CHECK first5cks=%s\n', mat2str(cks(1:min(5,end))', 6));
assert(isfinite(fpNorm) && isfinite(sum(cks)), 'Resume fingerprint non-finite — STOP.');

% --- pre-train patch audit: per-image vessel fraction AFTER resize ---
[~, ~, trFracs] = cropVessel256_resize(projRoot, trainIds);
fprintf('AUDIT train-resize vessel-frac min=%.4f med=%.4f p90=%.4f max=%.4f mean=%.4f\n', ...
    min(trFracs), median(trFracs), prctile(trFracs, 90), max(trFracs), mean(trFracs));
if mean(trFracs < 0.01) > 0.5
    fprintf('AUDIT-WARN: >50%% train images have <1%% vessel — flagged confound.\n');
end

seed = 42; rng(seed);
lr = 1e-4; bs = 4; epochs = 10; clipNorm = 1.0;
useGPU = (gpuDeviceCount > 0);
fprintf('OPT seed=%d lr=%.0e bs=%d clip=%.1f gpu=%d loss=vesselLoss_frozen\n', seed, lr, bs, clipNorm, useGPU);

nTrain = numel(trainIds);
itersPerEpoch = ceil(nTrain / bs);
nIter = itersPerEpoch * epochs;
lossLog = zeros(nIter, 1); nPreLog = zeros(nIter, 1);
clipLog = false(nIter, 1);
mA = []; vA = []; step = 0; it = 0;
over5 = 0;
evalRows = struct('epoch', {}, 'id', {}, 'dFix', {}, 'dBest', {}, 'bestThr', {}, ...
    'predFrac', {}, 'gtFrac', {}, 'fovFpFrac', {}, 'thinRecall', {});
csvPath = fullfile(outDir, 'train_ArmA.csv');
cf = fopen(csvPath, 'w');
fprintf(cf, 'iter,epoch,loss,nPre,clip,over5count\n');
t0 = tic;
for ep = 1:epochs
    order = randperm(nTrain);
    for b = 1:itersPerEpoch
        it = it + 1;
        idx = order((b-1)*bs+1:min(b*bs, nTrain));
        if numel(idx) < bs, idx = [idx, order(1:bs-numel(idx))]; end
        batchIds = trainIds(idx);
        [Xb, Tb] = cropVessel256_resize(projRoot, batchIds);
        if useGPU
            Xd = gpuArray(dlarray(Xb, 'SSCB'));
            Td = gpuArray(dlarray(Tb, 'SSCB'));
        else
            Xd = dlarray(Xb, 'SSCB');
            Td = dlarray(Tb, 'SSCB');
        end
        [loss, grads, ~, ~] = dlfeval(@vesselLoss_frozen, net, Xd, Td);
        lv = gather(extractdata(loss));
        assert(isfinite(lv), sprintf('Non-finite loss iter %d — STOP.', it));
        g2 = 0;
        for gi = 1:height(grads)
            gv = extractdata(grads.Value{gi});
            assert(all(isfinite(gv(:)), 'all'), sprintf('Non-finite grads iter %d — STOP.', it));
            g2 = g2 + sum(double(gv(:)).^2);
        end
        np = sqrt(g2);
        assert(isfinite(np) && np <= 1e6, sprintf('Grad-norm %.3g iter %d — STOP.', np, it));
        sc = min(1, clipNorm / (np + eps));
        grads = dlupdate(@(g) g*sc, grads);
        lossLog(it) = lv; nPreLog(it) = np; clipLog(it) = sc < 1;
        if lv > 5, over5 = over5 + 1; else, over5 = 0; end
        assert(over5 < 50, 'Loss>5 sustained 50 iters — STOP.');
        step = step + 1;
        [net.Learnables, mA, vA] = adamupdate(net.Learnables, grads, mA, vA, step, lr);
        fprintf(cf, '%d,%d,%.6f,%.4f,%d,%d\n', it, ep, lv, np, sc < 1, over5);
        fprintf('ARM-A iter=%d/%d ep=%d loss=%.4f nPre=%.2f clip=%d elapsed=%.1fmin\n', ...
            it, nIter, ep, lv, np, sc < 1, toc(t0)/60);
    end
    % per-epoch checkpoint (every epoch, not just best/final)
    ckFile = fullfile(outDir, sprintf('armA_epoch%d.mat', ep));
    save(ckFile, 'net', 'ep');
    % per-epoch per-image eval on fixed 4-val
    R = evalVesselEpoch(net, projRoot, valIds, useGPU, sprintf('ARM-A ep%d', ep));
    for k = 1:numel(R)
        evalRows(end+1).epoch = ep; %#ok<AGROW>
        evalRows(end).id = R(k).id; evalRows(end).dFix = R(k).dFix;
        evalRows(end).dBest = R(k).dBest; evalRows(end).bestThr = R(k).bestThr;
        evalRows(end).predFrac = R(k).predFrac; evalRows(end).gtFrac = R(k).gtFrac;
        evalRows(end).fovFpFrac = R(k).fovFpFrac; evalRows(end).thinRecall = R(k).thinRecall;
    end
    fprintf('ARM-A EPOCH %d DONE pooled dFix=%.4f dBest=%.4f\n', ep, mean([R.dFix]), mean([R.dBest]));
end
fclose(cf);
% eval CSV per image per epoch
ef = fopen(fullfile(outDir, 'eval_ArmA_perepoch.csv'), 'w');
fprintf(ef, 'epoch,id,dFix,dBest,bestThr,predFrac,gtFrac,fovFpFrac,thinRecall\n');
for i = 1:numel(evalRows)
    fprintf(ef, '%d,%d,%.5f,%.5f,%.2f,%.5f,%.5f,%.5f,%.5f\n', evalRows(i).epoch, ...
        evalRows(i).id, evalRows(i).dFix, evalRows(i).dBest, evalRows(i).bestThr, ...
        evalRows(i).predFrac, evalRows(i).gtFrac, evalRows(i).fovFpFrac, evalRows(i).thinRecall);
end
fclose(ef);
% best epoch by val Dice (not assumed epoch 10)
byEp = arrayfun(@(e) mean([evalRows([evalRows.epoch] == e).dFix]), 1:epochs);
[~, bestEp] = max(byEp);
fprintf('ARM-A DONE bestEp=%d pooledDfix=%.4f clipRate=%.3f\n', bestEp, byEp(bestEp), mean(clipLog));
% qualitative overlays at best epoch for 2 val images (37, 40)
netBest = net; % current net is epoch-10; reload best for overlays
B = load(fullfile(outDir, sprintf('armA_epoch%d.mat', bestEp)), 'net');
netBest = B.net;
for vid = [37 40]
    ipath = fullfile(projRoot, 'dataset/DRIVE/training/images', sprintf('%d_training.tif', vid));
    I = imread(ipath); [H, W, ~] = size(I);
    I256 = imresize(I, [256 256]);
    if useGPU, Xd = gpuArray(dlarray(single(I256), 'SSC')); else, Xd = dlarray(single(I256), 'SSC'); end
    Y = forward(netBest, Xd);
    P256 = single(extractdata(gather(Y(:, :, 2))));
    M = imresize(P256 >= 0.5, [H W], 'nearest');
    ov = labeloverlay(I, M, 'Transparency', 0.55);
    imwrite(ov, fullfile(outDir, sprintf('overlay_bestEp%d_id%d.png', bestEp, vid)));
end
save(fullfile(outDir, 'summary_ArmA.mat'), 'lossLog', 'nPreLog', 'clipLog', 'evalRows', ...
    'trainIds', 'valIds', 'fpNorm', 'bestEp');
diary off;
fprintf('ARM-A ALL_DONE outDir=%s\n', outDir);
