% trainVesselR2_ArmB — R2 FRESH rerun, Arm B (native stride-128).
% Same R2 amendment as Arm A: 160-iter budget, eval+ckpt every 40 iters
% (4 blocks), rebased gate (2x median-iters-1-10 sustained 50 + 1e6 absolute),
% CE/Dice logged per iter, loss math unchanged. Native 256 patches stride 128,
% FOV>=0.10 filter, 16/4 image split, test blind. Writes only to
% results/VES_P2B_R2_20260923/arm_native/.

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
addpath(fullfile(projRoot, 'segmentation/vessels/training_VES_P2B'));
addpath(fullfile(projRoot, 'segmentation/vessels'));
outDir = fullfile(projRoot, 'results/VES_P2B_R2_20260923/arm_native');
logPath = fullfile(outDir, 'console_R2_ArmB.log');
diary(logPath);
ps = 256; st = 128;
fprintf('R2-ARMB budget=160iters eval/ckpt every 40 (4 blocks) stride=%d rng=42 lr=1e-4 bs=4 clip=1.0\n', st);

trainIds = 21:36; valIds = 37:40;
S = load(fullfile(projRoot, 'results/vesselNet_weighted_5epoch.mat'), 'vesselNetWeightedTrained');
net = S.vesselNetWeightedTrained;
vals = net.Learnables.Value;
totN = 0; cks = zeros(numel(vals), 1);
for i = 1:numel(vals)
    v = double(gather(extractdata(vals{i})));
    totN = totN + sum(v(:).^2);
    cks(i) = sum(v(:));
end
fprintf('RESUME-CHECK fpNorm=%.6f ckSum=%.6f (expect 39.259579 / -1307.510348)\n', sqrt(totN), sum(cks));

nTrain = numel(trainIds);
Imgs = cell(nTrain, 1); Vessels = cell(nTrain, 1); FOVs = cell(nTrain, 1);
for i = 1:nTrain
    Imgs{i} = imread(fullfile(projRoot, 'dataset/DRIVE/training/images', sprintf('%d_training.tif', trainIds(i))));
    Vessels{i} = imread(fullfile(projRoot, 'dataset/DRIVE/training/1st_manual', sprintf('%d_manual1.gif', trainIds(i)))) > 0;
    FOVs{i} = imread(fullfile(projRoot, 'dataset/DRIVE/training/mask', sprintf('%d_training_mask.gif', trainIds(i)))) > 0;
end
[H, W, ~] = size(Imgs{1});
rs = unique([1:st:(H-ps), H-ps+1]); cs = unique([1:st:(W-ps), W-ps+1]);
P = struct('img', {}, 'r', {}, 'c', {}, 'vfrac', {}, 'ffrac', {});
for i = 1:nTrain
    for r = rs
        for c = cs
            P(end+1).img = i; %#ok<AGROW>
            P(end).r = r; P(end).c = c;
            P(end).vfrac = nnz(Vessels{i}(r:r+ps-1, c:c+ps-1))/ps^2;
            P(end).ffrac = nnz(FOVs{i}(r:r+ps-1, c:c+ps-1))/ps^2;
        end
    end
end
fprintf('PATCH-AUDIT total=%d kept(fov>=0.10)=%d vfrac med=%.4f max=%.4f\n', ...
    numel(P), nnz([P.ffrac] >= 0.10), median([P.vfrac]), max([P.vfrac]));
P = P([P.ffrac] >= 0.10);

seed = 42; rng(seed);
lr = 1e-4; bs = 4; clipNorm = 1.0;
useGPU = (gpuDeviceCount > 0);
nP = numel(P); budget = 160; block = 40;
fprintf('OPT patches=%d budget=%d blocks=4\n', nP, budget);
mA = []; vA = []; step = 0;
refWins = zeros(10, 1); refSet = false; refMed = NaN; overRel = 0;
cf = fopen(fullfile(outDir, 'train_R2_ArmB.csv'), 'w');
fprintf(cf, 'iter,loss,ce,dice,nPre,clip,overRel\n');
evalRows = struct('block', {}, 'id', {}, 'dFix', {}, 'dBest', {}, 'bestThr', {}, ...
    'predFrac', {}, 'gtFrac', {}, 'fovFpFrac', {}, 'thinRecall', {});
queue = [];
t0 = tic;
for it = 1:budget
    if numel(queue) < bs
        queue = [queue, randperm(nP)]; %#ok<AGROW>
    end
    take = queue(1:bs); queue(1:bs) = [];
    Xb = zeros(ps, ps, 3, bs, 'single');
    Tb = zeros(ps, ps, 2, bs, 'single');
    for j = 1:bs
        q = P(take(j));
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
    [loss, grads, ceP, diP] = dlfeval(@vesselLoss_frozen, net, Xd, Td);
    lv = gather(extractdata(loss));
    ce = gather(extractdata(ceP)); di = gather(extractdata(diP));
    assert(isfinite(lv), sprintf('Non-finite loss iter %d — STOP.', it));
    assert(lv <= 1e6, sprintf('Loss %.3g > 1e6 absolute backstop iter %d — STOP.', lv, it));
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
    if it <= 10
        refWins(it) = lv;
        if it == 10
            refMed = median(refWins);
            refSet = true;
            fprintf('GATE-REF median(iters1-10)=%.4f trip-level=%.4f\n', refMed, 2*refMed);
        end
    else
        if lv > 2*refMed, overRel = overRel + 1; else, overRel = 0; end
        assert(overRel < 50, 'Loss > 2x reference sustained 50 iters — STOP.');
    end
    step = step + 1;
    [net.Learnables, mA, vA] = adamupdate(net.Learnables, grads, mA, vA, step, lr);
    fprintf(cf, '%d,%.6f,%.6f,%.6f,%.4f,%d,%d\n', it, lv, ce, di, np, sc < 1, overRel);
    if mod(it, 20) == 0
        fprintf('R2-ARMB iter=%d/%d loss=%.4f (ce=%.3f di=%.3f) nPre=%.2f clip=%d elapsed=%.1fmin\n', ...
            it, budget, lv, ce, di, np, sc < 1, toc(t0)/60);
    end
    if mod(it, block) == 0
        bi = it / block;
        save(fullfile(outDir, sprintf('armB_block%d_iter%d.mat', bi, it)), 'net', 'it');
        R = evalVesselEpochNative(net, projRoot, valIds, useGPU, sprintf('R2-ARMB block%d', bi));
        for k = 1:numel(R)
            evalRows(end+1).block = bi; %#ok<AGROW>
            evalRows(end).id = R(k).id; evalRows(end).dFix = R(k).dFix;
            evalRows(end).dBest = R(k).dBest; evalRows(end).bestThr = R(k).bestThr;
            evalRows(end).predFrac = R(k).predFrac; evalRows(end).gtFrac = R(k).gtFrac;
            evalRows(end).fovFpFrac = R(k).fovFpFrac; evalRows(end).thinRecall = R(k).thinRecall;
        end
        fprintf('R2-ARMB BLOCK %d DONE pooled dFix=%.4f dBest=%.4f\n', bi, mean([R.dFix]), mean([R.dBest]));
    end
end
fclose(cf);
ef = fopen(fullfile(outDir, 'eval_R2_ArmB.csv'), 'w');
fprintf(ef, 'block,id,dFix,dBest,bestThr,predFrac,gtFrac,fovFpFrac,thinRecall\n');
for i = 1:numel(evalRows)
    fprintf(ef, '%d,%d,%.5f,%.5f,%.2f,%.5f,%.5f,%.5f,%.5f\n', evalRows(i).block, ...
        evalRows(i).id, evalRows(i).dFix, evalRows(i).dBest, evalRows(i).bestThr, ...
        evalRows(i).predFrac, evalRows(i).gtFrac, evalRows(i).fovFpFrac, evalRows(i).thinRecall);
end
fclose(ef);
byB = arrayfun(@(b) mean([evalRows([evalRows.block] == b).dFix]), 1:4);
[~, bestB] = max(byB);
fprintf('R2-ARMB DONE bestBlock=%d pooledDfix=%.4f\n', bestB, byB(bestB));
B2 = load(fullfile(outDir, sprintf('armB_block%d_iter%d.mat', bestB, bestB*block)), 'net');
netBest = B2.net;
for vid = [37 40]
    I = imread(fullfile(projRoot, 'dataset/DRIVE/training/images', sprintf('%d_training.tif', vid)));
    [Hh, Ww, ~] = size(I);
    rs2 = unique([1:st:(Hh-ps), Hh-ps+1]); cs2 = unique([1:st:(Ww-ps), Ww-ps+1]);
    acc = zeros(Hh, Ww, 'single'); cnt = zeros(Hh, Ww, 'single');
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
        fullfile(outDir, sprintf('overlay_bestB%d_id%d.png', bestB, vid)));
end
save(fullfile(outDir, 'summary_R2_ArmB.mat'), 'evalRows', 'bestB', 'refMed');
diary off;
fprintf('R2-ARMB ALL_DONE outDir=%s\n', outDir);
