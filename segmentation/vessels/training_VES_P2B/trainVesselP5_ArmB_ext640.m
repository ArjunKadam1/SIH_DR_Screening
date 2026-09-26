% trainVesselP5_ArmB_ext640 — pre-registered exposure extension: 160 → 640 iters.
% Same focal loss, same native data/split, same gates (original refMed kept),
% eval+ckpt every 40 (blocks 5-16). Resumes armB_block4_iter160.mat net.
% RNG: original run consumed ~3x randperm(256) refills (640 takes/256); replay
% rng(42) + discard 3x randperm(256) to approximate continuation (logged;
% exact sequence unrecoverable since rng state wasn't saved — uniform sampling
% makes this statistically equivalent, determinism noted as approximate).
% Adam moments: not saved in block ckpts — warm-restart fresh, step continues
% at 160 (same documented compromise as Arm A v1 extension). No overwrites:
% block5-16 mats/CSVs new; blocks 1-4 untouched. Purpose: falsify "needs steps".

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
addpath(fullfile(projRoot, 'segmentation/vessels/training_VES_P2B'));
addpath(fullfile(projRoot, 'segmentation/vessels'));
outDir = fullfile(projRoot, 'results/VES_P5_FOCAL_20260923/arm_native');
logPath = fullfile(outDir, 'console_P5_ArmB_ext640.log');
diary(logPath);
ps = 256; st = 128;
fprintf('P5-EXT 160->640 focal, blocks 5-16, same gates (refMed carried)\n');

trainIds = 21:36; valIds = 37:40;
B = load(fullfile(outDir, 'armB_block4_iter160.mat'), 'net', 'it');
net = B.net;
S0 = load(fullfile(outDir, 'summary_P5_ArmB.mat'), 'refMed');
refMed = S0.refMed;
fprintf('RESUME block4 iter=%d refMed carried=%.4f trip-level=%.4f\n', B.it, refMed, 2*refMed);
vals = net.Learnables.Value;
totN = 0;
for i = 1:numel(vals)
    v = double(gather(extractdata(vals{i})));
    totN = totN + sum(v(:).^2);
end
fprintf('RESUME-CHECK fpNorm=%.6f\n', sqrt(totN));

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
P = P([P.ffrac] >= 0.10);
nP = numel(P);

seed = 42; rng(seed);
for skip = 1:3, randperm(nP); end
fprintf('RNG replayed seed=42, discarded 3x randperm(%d) (~640 prior takes)\n', nP);
lr = 1e-4; bs = 4; clipNorm = 1.0;
useGPU = (gpuDeviceCount > 0);
budget = 640; block = 40;
mA = []; vA = []; step = 160; overRel = 0;
fprintf('OPT adamStep resumes at 160, moments fresh (warm-restart)\n');
cf = fopen(fullfile(outDir, 'train_P5_ArmB_ext640.csv'), 'w');
fprintf(cf, 'iter,loss,focal,dice,nPre,clip,overRel\n');
evalRows = struct('block', {}, 'id', {}, 'dFix', {}, 'dBest', {}, 'bestThr', {}, ...
    'predFrac', {}, 'gtFrac', {}, 'fovFpFrac', {}, 'thinRecall', {});
queue = [];
t0 = tic;
for it = 161:budget
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
    [loss, grads, foP, diP] = dlfeval(@vesselLoss_focal, net, Xd, Td);
    lv = gather(extractdata(loss));
    fo = gather(extractdata(foP)); di = gather(extractdata(diP));
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
    fprintf(cf, '%d,%.6f,%.6f,%.6f,%.4e,%d,%d\n', it, lv, fo, di, np, sc < 1, overRel);
    if mod(it, 40) == 0
        fprintf('P5-EXT iter=%d/%d loss=%.4f (focal=%.3f di=%.3f) nPre=%.4e clip=%d elapsed=%.1fmin\n', ...
            it, budget, lv, fo, di, np, sc < 1, toc(t0)/60);
    end
    if mod(it, block) == 0
        bi = it / block;
        save(fullfile(outDir, sprintf('armB_block%d_iter%d.mat', bi, it)), 'net', 'it');
        R = evalVesselEpochNative(net, projRoot, valIds, useGPU, sprintf('P5-EXT block%d', bi));
        for k = 1:numel(R)
            evalRows(end+1).block = bi; %#ok<AGROW>
            evalRows(end).id = R(k).id; evalRows(end).dFix = R(k).dFix;
            evalRows(end).dBest = R(k).dBest; evalRows(end).bestThr = R(k).bestThr;
            evalRows(end).predFrac = R(k).predFrac; evalRows(end).gtFrac = R(k).gtFrac;
            evalRows(end).fovFpFrac = R(k).fovFpFrac; evalRows(end).thinRecall = R(k).thinRecall;
        end
        fprintf('P5-EXT BLOCK %d DONE pooled dFix=%.4f dBest=%.4f sweeps=%.2f\n', ...
            bi, mean([R.dFix]), mean([R.dBest]), it*bs/nP);
    end
end
fclose(cf);
ef = fopen(fullfile(outDir, 'eval_P5_ArmB_ext640.csv'), 'w');
fprintf(ef, 'block,id,dFix,dBest,bestThr,predFrac,gtFrac,fovFpFrac,thinRecall\n');
for i = 1:numel(evalRows)
    fprintf(ef, '%d,%d,%.5f,%.5f,%.2f,%.5f,%.5f,%.5f,%.5f\n', evalRows(i).block, ...
        evalRows(i).id, evalRows(i).dFix, evalRows(i).dBest, evalRows(i).bestThr, ...
        evalRows(i).predFrac, evalRows(i).gtFrac, evalRows(i).fovFpFrac, evalRows(i).thinRecall);
end
fclose(ef);
byB = arrayfun(@(b) mean([evalRows([evalRows.block] == b).dFix]), 5:16);
[~, bi] = max(byB); bestB = bi + 4;
fprintf('P5-EXT DONE bestBlock(5-16)=%d pooledDfix=%.4f\n', bestB, byB(bi));
% closing probe vs probe0 (0.00309) for direction verdict
prE = 0;
for qp = 1:4
    q0 = P(qp);
    tile0 = im2single(Imgs{q0.img}(q0.r:q0.r+ps-1, q0.c:q0.c+ps-1, :));
    if useGPU, Xp = gpuArray(dlarray(tile0, 'SSC')); else, Xp = dlarray(tile0, 'SSC'); end
    Yp = forward(net, Xp);
    prE = prE + gather(extractdata(mean(Yp(:, :, 2), 'all')));
end
fprintf('TRIPWIRE-CLOSE probe: iter0=0.00309 iter10=0.00937 iter640=%.5f\n', prE/4);
save(fullfile(outDir, 'summary_P5_ArmB_ext640.mat'), 'evalRows', 'bestB');
diary off;
fprintf('P5-EXT ALL_DONE outDir=%s\n', outDir);
