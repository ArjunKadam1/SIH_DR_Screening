% trainVesselP6_AttB — P6 attention branch, native stride-128 data (frozen).
% Variable vs ALL prior arms: architecture (3 Oktay-style gates on skips,
% 69 layers, Fint=Cx/2). Init: attInit.mat warm-start (98.86% params copied,
% 18 fresh gate tensors); D1 PASSED (shape [256 256 2 4], grads reach all gates).
% Loss frozen (focal gamma=2 + Dice x5 — D2 holds by construction, same math).
% Budget 640 iters / eval+ckpt every 40 (16 blocks) to match P5-ext exposure.
% Same optimizer/gates/split/eval. Controls banked: P5-ext native 0.0004 flat
% over 640, P3-A resize 0.3735. Pre-registered: block dFix > 0.10 reopens line.
% Writes only to results/VES_P6_ATT_20260923/ (alongside attInit.mat).

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
addpath(fullfile(projRoot, 'segmentation/vessels/training_VES_P2B'));
addpath(fullfile(projRoot, 'segmentation/vessels'));
outDir = fullfile(projRoot, 'results/VES_P6_ATT_20260923');
logPath = fullfile(outDir, 'console_P6_AttB.log');
diary(logPath);
ps = 256; st = 128;
fprintf('P6-ATT arch=attention3gates loss=focal+dice5 budget=640 eval/ckpt every 40 stride=%d rng=42 lr=1e-4 bs=4 clip=1.0\n', st);

trainIds = 21:36; valIds = 37:40;
A = load(fullfile(outDir, 'attInit.mat'), 'attNet');
net = A.attNet;
% spot-check warm-start integrity vs plain checkpoint
SP = load(fullfile(projRoot, 'results/vesselNet_weighted_5epoch.mat'), 'vesselNetWeightedTrained');
pT = SP.vesselNetWeightedTrained.Learnables; aT = net.Learnables;
spot = 0;
for chk = ["Encoder-Stage-1-Conv-1/Weights", "Decoder-Stage-2-Conv-2/Bias", "encoderDecoderFinalConvLayer/Weights"]
    parts = split(chk, "/");
    ip = find(string(pT.Layer) == parts(1) & string(pT.Parameter) == parts(2), 1);
    ia = find(string(aT.Layer) == parts(1) & string(aT.Parameter) == parts(2), 1);
    Av = double(gather(extractdata(aT.Value{ia})));
    Pv = double(gather(extractdata(pT.Value{ip})));
    d = max(abs(Av(:) - Pv(:)));
    fprintf('SPOT-CHECK %s maxabsdiff=%.3g\n', chk, d);
    spot = max(spot, d);
end
assert(spot == 0, 'Warm-start copy corrupted — STOP.');
vals = net.Learnables.Value;
totN = 0;
for i = 1:numel(vals)
    v = double(gather(extractdata(vals{i})));
    totN = totN + sum(v(:).^2);
end
fprintf('RESUME-CHECK attNet fpNorm=%.6f layers=%d (expect ~41.16 / 69)\n', sqrt(totN), numel(net.Layers));

% NOTE: focal needs no W0 (no class weights); W0 block intentionally absent.
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
nP = numel(P); budget = 640; block = 40;
mA = []; vA = []; step = 0;
refWins = zeros(10, 1); refSet = false; refMed = NaN; overRel = 0;
cf = fopen(fullfile(outDir, 'train_P6_AttB.csv'), 'w');
fprintf(cf, 'iter,loss,focal,dice,nPre,clip,overRel\n');
evalRows = struct('block', {}, 'id', {}, 'dFix', {}, 'dBest', {}, 'bestThr', {}, ...
    'predFrac', {}, 'gtFrac', {}, 'fovFpFrac', {}, 'thinRecall', {});
queue = [];
% DIRECTION TRIPWIRE probe: fixed first-4-patch mean vessel-P at iter0
probe0 = 0;
for qp = 1:4
    q0 = P(qp);
    tile0 = im2single(Imgs{q0.img}(q0.r:q0.r+ps-1, q0.c:q0.c+ps-1, :));
    if useGPU, Xp0 = gpuArray(dlarray(tile0, 'SSC')); else, Xp0 = dlarray(tile0, 'SSC'); end
    Yp0 = forward(net, Xp0);
    probe0 = probe0 + gather(extractdata(mean(Yp0(:, :, 2), 'all')));
end
probe0 = probe0 / 4;
fprintf('TRIPWIRE probe0 vessel-P mean=%.5f (fixed patches 1-4)\n', probe0);
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
    fprintf(cf, '%d,%.6f,%.6f,%.6f,%.4e,%d,%d\n', it, lv, fo, di, np, sc < 1, overRel);
    if mod(it, 20) == 0
        fprintf('P6-ATT iter=%d/%d loss=%.4f (focal=%.3f di=%.3f) nPre=%.4e clip=%d elapsed=%.1fmin\n', ...
            it, budget, lv, fo, di, np, sc < 1, toc(t0)/60);
    end
    if it == 10
        % DIRECTION TRIPWIRE (diagnostic only): fixed 4-patch probe frac now vs iter0
        pr1 = 0;
        for qp = 1:4
            q0 = P(qp);
            tile0 = im2single(Imgs{q0.img}(q0.r:q0.r+ps-1, q0.c:q0.c+ps-1, :));
            if useGPU, Xp = gpuArray(dlarray(tile0, 'SSC')); else, Xp = dlarray(tile0, 'SSC'); end
            Yp = forward(net, Xp);
            pr1 = pr1 + gather(extractdata(mean(Yp(:, :, 2), 'all')));
        end
        pr1 = pr1 / 4;
        fprintf('TRIPWIRE probe vessel-P mean: iter0=%.5f iter10=%.5f -> %s\n', ...
            probe0, pr1, string(pr1 >= probe0));
        if pr1 < probe0
            fprintf('TRIPWIRE-FLAG: predictions collapsing toward zero.\n');
        end
    end
    if mod(it, block) == 0
        bi = it / block;
        save(fullfile(outDir, sprintf('attB_block%d_iter%d.mat', bi, it)), 'net', 'it');
        R = evalVesselEpochNative(net, projRoot, valIds, useGPU, sprintf('P6-ATT block%d', bi));
        for k = 1:numel(R)
            evalRows(end+1).block = bi; %#ok<AGROW>
            evalRows(end).id = R(k).id; evalRows(end).dFix = R(k).dFix;
            evalRows(end).dBest = R(k).dBest; evalRows(end).bestThr = R(k).bestThr;
            evalRows(end).predFrac = R(k).predFrac; evalRows(end).gtFrac = R(k).gtFrac;
            evalRows(end).fovFpFrac = R(k).fovFpFrac; evalRows(end).thinRecall = R(k).thinRecall;
        end
        fprintf('P6-ATT BLOCK %d DONE pooled dFix=%.4f dBest=%.4f sweeps=%.2f\n', ...
            bi, mean([R.dFix]), mean([R.dBest]), it*bs/nP);
    end
end
fclose(cf);
ef = fopen(fullfile(outDir, 'eval_P6_AttB.csv'), 'w');
fprintf(ef, 'block,id,dFix,dBest,bestThr,predFrac,gtFrac,fovFpFrac,thinRecall\n');
for i = 1:numel(evalRows)
    fprintf(ef, '%d,%d,%.5f,%.5f,%.2f,%.5f,%.5f,%.5f,%.5f\n', evalRows(i).block, ...
        evalRows(i).id, evalRows(i).dFix, evalRows(i).dBest, evalRows(i).bestThr, ...
        evalRows(i).predFrac, evalRows(i).gtFrac, evalRows(i).fovFpFrac, evalRows(i).thinRecall);
end
fclose(ef);
byB = arrayfun(@(b) mean([evalRows([evalRows.block] == b).dFix]), 1:16);
[~, bestB] = max(byB);
fprintf('P6-ATT DONE bestBlock=%d pooledDfix=%.4f\n', bestB, byB(bestB));
B2 = load(fullfile(outDir, sprintf('attB_block%d_iter%d.mat', bestB, bestB*block)), 'net');
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
        fullfile(outDir, sprintf('overlay_attBestB%d_id%d.png', bestB, vid)));
end
save(fullfile(outDir, 'summary_P6_AttB.mat'), 'evalRows', 'bestB', 'refMed', 'probe0');
diary off;
fprintf('P6-ATT ALL_DONE outDir=%s\n', outDir);
