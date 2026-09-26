% trainVesselArmB_native — VES P2B Arm B (test: native resolution, matched training).
% Resumes vesselNetWeightedTrained byte-identical. Native 256x256 patches
% stride 128 from 584x565 DRIVE (no resize), FOV-aware filtered (fovFrac>=0.10),
% 16 train / 4 val image-level split (21-36 / 37-40), test 01-20 blind.
% Frozen: Adam 1e-4, batch 4 patches, clip 1.0, rng(42), vesselLoss_frozen,
% aug none, 10 epochs, per-epoch ckpts. Gates: NaN/Inf STOP, norm>1e6 STOP,
% loss>5 sustained 50 iters STOP.

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
addpath(fullfile(projRoot, 'segmentation/vessels/training_VES_P2B'));
addpath(fullfile(projRoot, 'segmentation/vessels'));
stamp = datestr(now, 'yyyymmdd_HHMMSS');
outDir = fullfile(projRoot, 'results', ['VES_P2B_256native_' stamp]);
mkdir(outDir);
logPath = fullfile(outDir, 'console_ArmB.log');
diary(logPath);
ps = 256; st = 128;
fprintf('ARM-B NATIVE stride=%d rng=42 lr=1e-4 bs=4patches clip=1.0 epochs=10\n', st);

trainIds = 21:36; valIds = 37:40;
fprintf('SPLIT train=%s val=%s (image-level, test 01-20 blind)\n', ...
    mat2str(trainIds), mat2str(valIds));

% --- byte-identical resume check (same fingerprint as Arm A) ---
S = load(fullfile(projRoot, 'results/vesselNet_weighted_5epoch.mat'), 'vesselNetWeightedTrained');
net = S.vesselNetWeightedTrained;
vals = net.Learnables.Value;
totN = 0; cks = zeros(numel(vals), 1);
for i = 1:numel(vals)
    v = double(gather(extractdata(vals{i})));
    totN = totN + sum(v(:).^2);
    cks(i) = sum(v(:));
end
fprintf('RESUME-CHECK var=vesselNetWeightedTrained layers=%d fpNorm=%.6f ckSum=%.6f nTensors=%d\n', ...
    numel(net.Layers), sqrt(totN), sum(cks), numel(vals));
fprintf('RESUME-CHECK first5cks=%s (must match Arm A: fpNorm=39.259579 ckSum=-1307.510348)\n', ...
    mat2str(cks(1:min(5,end))', 6));

% --- preload train images + build native patch table ---
nTrain = numel(trainIds);
Imgs = cell(nTrain, 1); Vessels = cell(nTrain, 1);
for i = 1:nTrain
    Imgs{i} = imread(fullfile(projRoot, 'dataset/DRIVE/training/images', sprintf('%d_training.tif', trainIds(i))));
    Vessels{i} = imread(fullfile(projRoot, 'dataset/DRIVE/training/1st_manual', sprintf('%d_manual1.gif', trainIds(i)))) > 0;
    F = imread(fullfile(projRoot, 'dataset/DRIVE/training/mask', sprintf('%d_training_mask.gif', trainIds(i)))) > 0;
    if i == 1, FOVs = cell(nTrain, 1); end
    FOVs{i} = F;
end
[H, W, ~] = size(Imgs{1});
rs = unique([1:st:(H-ps), H-ps+1]); cs = unique([1:st:(W-ps), W-ps+1]);
fprintf('GRID H=%d W=%d ps=%d st=%d rs=%s cs=%s patches/img=%d\n', ...
    H, W, ps, st, mat2str(rs), mat2str(cs), numel(rs)*numel(cs));
P = struct('img', {}, 'r', {}, 'c', {}, 'vfrac', {}, 'ffrac', {});
for i = 1:nTrain
    for r = rs
        for c = cs
            Vp = Vessels{i}(r:r+ps-1, c:c+ps-1);
            Fp = FOVs{i}(r:r+ps-1, c:c+ps-1);
            P(end+1).img = i; %#ok<AGROW>
            P(end).r = r; P(end).c = c;
            P(end).vfrac = nnz(Vp)/numel(Vp);
            P(end).ffrac = nnz(Fp)/numel(Fp);
        end
    end
end
fprintf('PATCH-AUDIT total=%d vfrac min=%.4f med=%.4f p90=%.4f max=%.4f mean=%.4f\n', ...
    numel(P), min([P.vfrac]), median([P.vfrac]), prctile([P.vfrac], 90), ...
    max([P.vfrac]), mean([P.vfrac]));
fprintf('PATCH-AUDIT fovfrac<0.10 patches=%d/%d (%.1f%%) — EXCLUDED as non-retina corners\n', ...
    nnz([P.ffrac] < 0.10), numel(P), 100*nnz([P.ffrac] < 0.10)/numel(P));
P = P([P.ffrac] >= 0.10);
fprintf('PATCH-AUDIT kept=%d vfrac min=%.4f med=%.4f p90=%.4f max=%.4f frac<0.01: %d (%.1f%%)\n', ...
    numel(P), min([P.vfrac]), median([P.vfrac]), prctile([P.vfrac], 90), ...
    max([P.vfrac]), nnz([P.vfrac] < 0.01), 100*nnz([P.vfrac] < 0.01)/numel(P));
assert(~isempty(P), 'No patches after FOV filter — STOP.');

seed = 42; rng(seed);
lr = 1e-4; bs = 4; epochs = 10; clipNorm = 1.0;
useGPU = (gpuDeviceCount > 0);
nP = numel(P);
itersPerEpoch = ceil(nP / bs);
nIter = itersPerEpoch * epochs;
fprintf('OPT seed=%d lr=1e-4 bs=%d clip=1.0 gpu=%d loss=vesselLoss_frozen patches=%d iters/ep=%d total=%d\n', ...
    seed, bs, useGPU, nP, itersPerEpoch, nIter);
mA = []; vA = []; step = 0; it = 0; over5 = 0;
cf = fopen(fullfile(outDir, 'train_ArmB.csv'), 'w');
fprintf(cf, 'iter,epoch,loss,nPre,clip,over5count\n');
evalRows = struct('epoch', {}, 'id', {}, 'dFix', {}, 'dBest', {}, 'bestThr', {}, ...
    'predFrac', {}, 'gtFrac', {}, 'fovFpFrac', {}, 'thinRecall', {});
t0 = tic;
for ep = 1:epochs
    order = randperm(nP);
    for b = 1:itersPerEpoch
        it = it + 1;
        sidx = order((b-1)*bs+1:min(b*bs, nP));
        if numel(sidx) < bs, sidx = [sidx, order(1:bs-numel(sidx))]; end
        Xb = zeros(ps, ps, 3, bs, 'single');
        Tb = zeros(ps, ps, 2, bs, 'single');
        for j = 1:bs
            q = P(sidx(j));
            Ip = Imgs{q.img}(q.r:q.r+ps-1, q.c:q.c+ps-1, :);
            Vp = Vessels{q.img}(q.r:q.r+ps-1, q.c:q.c+ps-1);
            Xb(:, :, :, j) = im2single(Ip);
            Tb(:, :, 1, j) = single(~Vp);
            Tb(:, :, 2, j) = single(Vp);
        end
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
        if lv > 5, over5 = over5 + 1; else, over5 = 0; end
        assert(over5 < 50, 'Loss>5 sustained 50 iters — STOP.');
        step = step + 1;
        [net.Learnables, mA, vA] = adamupdate(net.Learnables, grads, mA, vA, step, lr);
        fprintf(cf, '%d,%d,%.6f,%.4f,%d,%d\n', it, ep, lv, np, sc < 1, over5);
        if mod(it, 20) == 0
            fprintf('ARM-B iter=%d/%d ep=%d loss=%.4f nPre=%.2f clip=%d elapsed=%.1fmin\n', ...
                it, nIter, ep, lv, np, sc < 1, toc(t0)/60);
        end
    end
    save(fullfile(outDir, sprintf('armB_epoch%d.mat', ep)), 'net', 'ep');
    R = evalVesselEpochNative(net, projRoot, valIds, useGPU, sprintf('ARM-B ep%d', ep));
    for k = 1:numel(R)
        evalRows(end+1).epoch = ep; %#ok<AGROW>
        evalRows(end).id = R(k).id; evalRows(end).dFix = R(k).dFix;
        evalRows(end).dBest = R(k).dBest; evalRows(end).bestThr = R(k).bestThr;
        evalRows(end).predFrac = R(k).predFrac; evalRows(end).gtFrac = R(k).gtFrac;
        evalRows(end).fovFpFrac = R(k).fovFpFrac; evalRows(end).thinRecall = R(k).thinRecall;
    end
    fprintf('ARM-B EPOCH %d DONE pooled dFix=%.4f dBest=%.4f\n', ep, mean([R.dFix]), mean([R.dBest]));
end
fclose(cf);
ef = fopen(fullfile(outDir, 'eval_ArmB_perepoch.csv'), 'w');
fprintf(ef, 'epoch,id,dFix,dBest,bestThr,predFrac,gtFrac,fovFpFrac,thinRecall\n');
for i = 1:numel(evalRows)
    fprintf(ef, '%d,%d,%.5f,%.5f,%.2f,%.5f,%.5f,%.5f,%.5f\n', evalRows(i).epoch, ...
        evalRows(i).id, evalRows(i).dFix, evalRows(i).dBest, evalRows(i).bestThr, ...
        evalRows(i).predFrac, evalRows(i).gtFrac, evalRows(i).fovFpFrac, evalRows(i).thinRecall);
end
fclose(ef);
byEp = arrayfun(@(e) mean([evalRows([evalRows.epoch] == e).dFix]), 1:epochs);
[~, bestEp] = max(byEp);
fprintf('ARM-B DONE bestEp=%d pooledDfix=%.4f\n', bestEp, byEp(bestEp));
B2 = load(fullfile(outDir, sprintf('armB_epoch%d.mat', bestEp)), 'net');
netBest = B2.net;
% overlays at best epoch: reuse native tiled inference for 2 val images
for vid = [37 40]
    I = imread(fullfile(projRoot, 'dataset/DRIVE/training/images', sprintf('%d_training.tif', vid)));
    [H, W, ~] = size(I);
    rs2 = unique([1:st:(H-ps), H-ps+1]); cs2 = unique([1:st:(W-ps), W-ps+1]);
    acc = zeros(H, W, 'single'); cnt = zeros(H, W, 'single');
    for r = rs2
        for c = cs2
            tile = im2single(I(r:r+ps-1, c:c+ps-1, :));
            if useGPU, Xd = gpuArray(dlarray(tile, 'SSC')); else, Xd = dlarray(tile, 'SSC'); end
            Y = forward(netBest, Xd);
            Pv = single(extractdata(gather(Y(:, :, 2))));
            acc(r:r+ps-1, c:c+ps-1) = acc(r:r+ps-1, c:c+ps-1) + Pv;
            cnt(r:r+ps-1, c:c+ps-1) = cnt(r:r+ps-1, c:c+ps-1) + 1;
        end
    end
    M = (acc ./ max(cnt, 1)) >= 0.5;
    imwrite(labeloverlay(I, M, 'Transparency', 0.55), ...
        fullfile(outDir, sprintf('overlay_bestEp%d_id%d.png', bestEp, vid)));
end
save(fullfile(outDir, 'summary_ArmB.mat'), 'evalRows', 'bestEp');
diary off;
fprintf('ARM-B ALL_DONE outDir=%s\n', outDir);
