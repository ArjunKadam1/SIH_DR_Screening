% trainVesselArmA_resize_ext20 — extend Arm A (resize control) epochs 11-20.
% Resumes armA_epoch10.mat net. Same frozen settings: Adam 1e-4, batch 4,
% clip 1.0, vesselLoss_frozen, same 16/4 split, aug none. Gates unchanged.
% Determinism: rng(42) replayed + first 10 epoch orders discarded so ep11+
% order continues the original sequence. Adam moments were not saved per
% epoch in Arm A v1 — moments restart fresh with step counter continuing
% at 40 (documented warm-restart, logged below). No overwrites: new
% epoch11-20 mats/CSVs only; epoch1-10 files untouched.

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
addpath(fullfile(projRoot, 'segmentation/vessels/training_VES_P2B'));
addpath(fullfile(projRoot, 'segmentation/vessels'));
outDir = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening/results/VES_P2B_256resize_20260923_223746';
logPath = fullfile(outDir, 'console_ArmA_ext11_20.log');
diary(logPath);
fprintf('ARM-A-EXT epochs 11-20 from armA_epoch10.mat, frozen settings identical\n');

trainIds = 21:36; valIds = 37:40;
B = load(fullfile(outDir, 'armA_epoch10.mat'), 'net');
net = B.net;
vals = net.Learnables.Value;
totN = 0;
for i = 1:numel(vals)
    v = double(gather(extractdata(vals{i})));
    totN = totN + sum(v(:).^2);
end
fprintf('RESUME-CHECK ep10 fpNorm=%.6f layers=%d\n', sqrt(totN), numel(net.Layers));

% deterministic order continuation: replay rng(42) + discard first 10 epochs
seed = 42; rng(seed);
nTrain = numel(trainIds);
for skip = 1:10, randperm(nTrain); end
fprintf('RNG replayed seed=42, skipped 10 epoch orders; ep11 continues sequence\n');

lr = 1e-4; bs = 4; clipNorm = 1.0;
useGPU = (gpuDeviceCount > 0);
itersPerEpoch = ceil(nTrain / bs);
mA = []; vA = []; step = 40; % continue Adam step counter; moments fresh (warm-restart)
fprintf('OPT lr=1e-4 bs=4 clip=1.0 gpu=%d adamStep resumes at 40, moments fresh\n', useGPU);
it = 40; over5 = 0;
cf = fopen(fullfile(outDir, 'train_ArmA_ext11_20.csv'), 'w');
fprintf(cf, 'iter,epoch,loss,nPre,clip,over5count\n');
evalRows = struct('epoch', {}, 'id', {}, 'dFix', {}, 'dBest', {}, 'bestThr', {}, ...
    'predFrac', {}, 'gtFrac', {}, 'fovFpFrac', {}, 'thinRecall', {});
t0 = tic;
for ep = 11:20
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
        if lv > 5, over5 = over5 + 1; else, over5 = 0; end
        assert(over5 < 50, 'Loss>5 sustained 50 iters — STOP.');
        step = step + 1;
        [net.Learnables, mA, vA] = adamupdate(net.Learnables, grads, mA, vA, step, lr);
        fprintf(cf, '%d,%d,%.6f,%.4f,%d,%d\n', it, ep, lv, np, sc < 1, over5);
        fprintf('ARM-A iter=%d/80 ep=%d loss=%.4f nPre=%.3f clip=%d elapsed=%.1fmin\n', ...
            it, ep, lv, np, sc < 1, toc(t0)/60);
    end
    save(fullfile(outDir, sprintf('armA_epoch%d.mat', ep)), 'net', 'ep');
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
ef = fopen(fullfile(outDir, 'eval_ArmA_ext11_20.csv'), 'w');
fprintf(ef, 'epoch,id,dFix,dBest,bestThr,predFrac,gtFrac,fovFpFrac,thinRecall\n');
for i = 1:numel(evalRows)
    fprintf(ef, '%d,%d,%.5f,%.5f,%.2f,%.5f,%.5f,%.5f,%.5f\n', evalRows(i).epoch, ...
        evalRows(i).id, evalRows(i).dFix, evalRows(i).dBest, evalRows(i).bestThr, ...
        evalRows(i).predFrac, evalRows(i).gtFrac, evalRows(i).fovFpFrac, evalRows(i).thinRecall);
end
fclose(ef);
byEp = arrayfun(@(e) mean([evalRows([evalRows.epoch] == e).dFix]), 11:20);
[~, bi] = max(byEp); bestEp = bi + 10;
fprintf('ARM-A-EXT DONE bestEp(11-20)=%d pooledDfix=%.4f\n', bestEp, byEp(bi));
B2 = load(fullfile(outDir, sprintf('armA_epoch%d.mat', bestEp)), 'net');
netBest = B2.net;
for vid = [37 40]
    ipath = fullfile(projRoot, 'dataset/DRIVE/training/images', sprintf('%d_training.tif', vid));
    I = imread(ipath); [H, W, ~] = size(I);
    I256 = imresize(I, [256 256]);
    if useGPU, Xd = gpuArray(dlarray(single(I256), 'SSC')); else, Xd = dlarray(single(I256), 'SSC'); end
    Y = forward(netBest, Xd);
    P256 = single(extractdata(gather(Y(:, :, 2))));
    M = imresize(P256 >= 0.5, [H W], 'nearest');
    imwrite(labeloverlay(I, M, 'Transparency', 0.55), ...
        fullfile(outDir, sprintf('overlay_extBestEp%d_id%d.png', bestEp, vid)));
end
save(fullfile(outDir, 'summary_ArmA_ext11_20.mat'), 'evalRows', 'bestEp');
diary off;
fprintf('ARM-A-EXT ALL_DONE outDir=%s\n', outDir);
