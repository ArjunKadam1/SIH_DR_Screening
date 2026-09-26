% trainVesselP3A_ext320 — NEW BRANCH extension of P3-A (block 4 → +160 iters).
% Methodology identical to P3-A (plan-spec, user-approved): normalized loss
% with W0 recomputed by the identical code path (must log 10.4785 or STOP),
% 16/4 image split, batch 4, Adam 1e-4, clip 1.0, aug none, test blind,
% eval+ckpt every 40 (blocks 5-8), gates with refMed CARRIED from P3-A
% summary (2.2046) + 1e6 absolute + NaN/Inf + norm>1e6.
% Resume: armA_block4_iter160 net; rng(42) replayed + 10x randperm(16)
% discarded (160 prior takes); Adam moments fresh, step continues at 160.
% New branch: writes ONLY to results/VES_P3A_EXT_20260923/arm_resize/ with
% ext-block names. Old P3-A dir is read-source only, never written.

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
addpath(fullfile(projRoot, 'segmentation/vessels/training_VES_P2B'));
addpath(fullfile(projRoot, 'segmentation/vessels'));
srcDir = fullfile(projRoot, 'results/VES_P3_NORM_20260923/arm_resize');
outDir = fullfile(projRoot, 'results/VES_P3A_EXT_20260923/arm_resize');
logPath = fullfile(outDir, 'console_P3A_ext320.log');
diary(logPath);
fprintf('P3A-EXT NEW-BRANCH from block4 +160 (blocks 5-8) same methodology\n');

trainIds = 21:36; valIds = 37:40;
B = load(fullfile(srcDir, 'armA_block4_iter160.mat'), 'net');
net = B.net;
S0 = load(fullfile(srcDir, 'summary_P3_ArmA.mat'), 'refMed');
refMed = S0.refMed;
fprintf('RESUME block4 refMed carried=%.4f trip-level=%.4f\n', refMed, 2*refMed);
vals = net.Learnables.Value;
totN = 0;
for i = 1:numel(vals)
    v = double(gather(extractdata(vals{i})));
    totN = totN + sum(v(:).^2);
end
fprintf('RESUME-CHECK fpNorm=%.6f layers=%d\n', sqrt(totN), numel(net.Layers));

% W0 by the identical code path — mismatch vs 10.4785 is a STOP
[~, ~, trFracs] = cropVessel256_resize(projRoot, trainIds);
vMean = mean(trFracs);
W0 = (1 - vMean) / max(vMean, eps);
fprintf('W0 recomputed=%.4f (expect 10.4785)\n', W0);
assert(abs(W0 - 10.4785) < 1e-3, 'W0 mismatch — STOP.');

seed = 42; rng(seed);
nTrain = numel(trainIds);
for skip = 1:10, randperm(nTrain); end
fprintf('RNG replayed seed=42, discarded 10x randperm(%d)\n', nTrain);
lr = 1e-4; bs = 4; clipNorm = 1.0;
useGPU = (gpuDeviceCount > 0);
budget = 320; block = 40;
mA = []; vA = []; step = 160; overRel = 0;
fprintf('OPT lr=1e-4 bs=4 clip=1.0 gpu=%d adamStep resumes at 160, moments fresh\n', useGPU);
cf = fopen(fullfile(outDir, 'train_P3A_ext320.csv'), 'w');
fprintf(cf, 'iter,loss,ce,dice,wV,nPre,clip,overRel\n');
evalRows = struct('block', {}, 'id', {}, 'dFix', {}, 'dBest', {}, 'bestThr', {}, ...
    'predFrac', {}, 'gtFrac', {}, 'fovFpFrac', {}, 'thinRecall', {});
queue = [];
t0 = tic;
for it = 161:budget
    if numel(queue) < bs
        queue = [queue, randperm(nTrain)]; %#ok<AGROW>
    end
    take = queue(1:bs); queue(1:bs) = [];
    batchIds = trainIds(take);
    [Xb, Tb] = cropVessel256_resize(projRoot, batchIds);
    if useGPU
        Xd = gpuArray(dlarray(Xb, 'SSCB'));
        Td = gpuArray(dlarray(Tb, 'SSCB'));
    else
        Xd = dlarray(Xb, 'SSCB');
        Td = dlarray(Tb, 'SSCB');
    end
    [loss, grads, ceP, diP, wV] = dlfeval(@vesselLoss_norm, net, Xd, Td, W0);
    lv = gather(extractdata(loss));
    ce = gather(extractdata(ceP)); di = gather(extractdata(diP));
    wv = gather(extractdata(wV)); if ~isscalar(wv), wv = double(wv); wv = wv(1); end
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
    if lv > 2*refMed, overRel = overRel + 1; else, overRel = 0; end
    assert(overRel < 50, 'Loss > 2x reference sustained 50 iters — STOP.');
    step = step + 1;
    [net.Learnables, mA, vA] = adamupdate(net.Learnables, grads, mA, vA, step, lr);
    fprintf(cf, '%d,%.6f,%.6f,%.6f,%.4f,%.4e,%d,%d\n', it, lv, ce, di, wv, np, sc < 1, overRel);
    if mod(it, 40) == 0
        fprintf('P3A-EXT iter=%d/%d loss=%.4f (ce=%.3f di=%.3f) nPre=%.4e clip=%d elapsed=%.1fmin\n', ...
            it, budget, lv, ce, di, np, sc < 1, toc(t0)/60);
    end
    if mod(it, block) == 0
        bi = it / block;
        save(fullfile(outDir, sprintf('extA_block%d_iter%d.mat', bi, it)), 'net', 'it');
        R = evalVesselEpoch(net, projRoot, valIds, useGPU, sprintf('P3A-EXT block%d', bi));
        for k = 1:numel(R)
            evalRows(end+1).block = bi; %#ok<AGROW>
            evalRows(end).id = R(k).id; evalRows(end).dFix = R(k).dFix;
            evalRows(end).dBest = R(k).dBest; evalRows(end).bestThr = R(k).bestThr;
            evalRows(end).predFrac = R(k).predFrac; evalRows(end).gtFrac = R(k).gtFrac;
            evalRows(end).fovFpFrac = R(k).fovFpFrac; evalRows(end).thinRecall = R(k).thinRecall;
        end
        fprintf('P3A-EXT BLOCK %d DONE pooled dFix=%.4f dBest=%.4f sweeps=%.1f\n', ...
            bi, mean([R.dFix]), mean([R.dBest]), it*bs/nTrain);
    end
end
fclose(cf);
ef = fopen(fullfile(outDir, 'eval_P3A_ext320.csv'), 'w');
fprintf(ef, 'block,id,dFix,dBest,bestThr,predFrac,gtFrac,fovFpFrac,thinRecall\n');
for i = 1:numel(evalRows)
    fprintf(ef, '%d,%d,%.5f,%.5f,%.2f,%.5f,%.5f,%.5f,%.5f\n', evalRows(i).block, ...
        evalRows(i).id, evalRows(i).dFix, evalRows(i).dBest, evalRows(i).bestThr, ...
        evalRows(i).predFrac, evalRows(i).gtFrac, evalRows(i).fovFpFrac, evalRows(i).thinRecall);
end
fclose(ef);
byB = arrayfun(@(b) mean([evalRows([evalRows.block] == b).dFix]), 5:8);
[~, bi] = max(byB); bestB = bi + 4;
fprintf('P3A-EXT DONE bestBlock(5-8)=%d pooledDfix=%.4f (P3-A block2 0.3735)\n', bestB, byB(bi));
B2 = load(fullfile(outDir, sprintf('extA_block%d_iter%d.mat', bestB, bestB*block)), 'net');
netBest = B2.net;
for vid = [37 40]
    I = imread(fullfile(projRoot, 'dataset/DRIVE/training/images', sprintf('%d_training.tif', vid)));
    [H, W, ~] = size(I);
    I256 = imresize(I, [256 256]);
    if useGPU, Xd = gpuArray(dlarray(single(I256), 'SSC')); else, Xd = dlarray(single(I256), 'SSC'); end
    Y = forward(netBest, Xd);
    M = imresize(single(extractdata(gather(Y(:, :, 2)))) >= 0.5, [H W], 'nearest');
    imwrite(labeloverlay(I, M, 'Transparency', 0.55), ...
        fullfile(outDir, sprintf('overlay_extBestB%d_id%d.png', bestB, vid)));
end
save(fullfile(outDir, 'summary_P3A_ext320.mat'), 'evalRows', 'bestB', 'refMed', 'W0');
diary off;
fprintf('P3A-EXT ALL_DONE outDir=%s (old P3-A dir untouched)\n', outDir);
