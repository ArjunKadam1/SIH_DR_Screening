% trainVesselP7_AttResize — P7: attention net (from P3-A block2 warm-start) on
% RESIZE protocol. Same graft as P6; loss frozen focal+dice5 (D2 holds by
% construction); data = resize-256 16 imgs bs=4 (same as P3-A); budget 640 /
% eval+ckpt every 40 (16 blocks) to match P6 exposure; Adam 1e-4 clip 1.0
% rng42 fresh, moments fresh step 0; gates (2x median-iters-1-10/50 + 1e6 abs +
% NaN/Inf + norm>1e6); 16/4 split, test blind. Direction probe (fixed 4 imgs
% as 256-resize batch) at iter0/10. Expectation logged: fresh gates output
% ~0.5, halving skips — initial Dice likely DIPS below P3-A 0.3735; trajectory
% is the verdict. Controls banked: P3-A 0.3735, P6-native best 0.045.
% Writes only to results/VES_P7_ATTRESIZE_20260923/.

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
addpath(fullfile(projRoot, 'segmentation/vessels/training_VES_P2B'));
addpath(fullfile(projRoot, 'segmentation/vessels'));
outDir = fullfile(projRoot, 'results/VES_P7_ATTRESIZE_20260923');
logPath = fullfile(outDir, 'console_P7_AttResize.log');
diary(logPath);
fprintf('P7-ATTRESIZE base=P3A-block2 loss=focal+dice5 budget=640 eval/ckpt every 40 rng=42 lr=1e-4 bs=4 clip=1.0\n');

trainIds = 21:36; valIds = 37:40;
A = load(fullfile(outDir, 'attInitP7.mat'), 'attNet');
net = A.attNet;
vals = net.Learnables.Value;
totN = 0;
for i = 1:numel(vals)
    v = double(gather(extractdata(vals{i})));
    totN = totN + sum(v(:).^2);
end
fprintf('RESUME-CHECK attInitP7 fpNorm=%.6f layers=%d\n', sqrt(totN), numel(net.Layers));

seed = 42; rng(seed);
lr = 1e-4; bs = 4; clipNorm = 1.0;
useGPU = (gpuDeviceCount > 0);
budget = 640; block = 40;
mA = []; vA = []; step = 0;
refWins = zeros(10, 1); refSet = false; refMed = NaN; overRel = 0;
cf = fopen(fullfile(outDir, 'train_P7.csv'), 'w');
fprintf(cf, 'iter,loss,focal,dice,nPre,clip,overRel\n');
evalRows = struct('block', {}, 'id', {}, 'dFix', {}, 'dBest', {}, 'bestThr', {}, ...
    'predFrac', {}, 'gtFrac', {}, 'fovFpFrac', {}, 'thinRecall', {});
queue = [];
% direction probe: fixed first-4 train images as resize batch, mean vessel-P
[PXb, ~] = cropVessel256_resize(projRoot, trainIds(1:4));
if useGPU, Xp0 = gpuArray(dlarray(PXb, 'SSCB')); else, Xp0 = dlarray(PXb, 'SSCB'); end
Yp0 = forward(net, Xp0);
probe0 = gather(extractdata(mean(Yp0(:, :, 2), 'all')));
fprintf('TRIPWIRE probe0 vessel-P mean=%.5f (expect ~half of P3-A state: fresh gates)\n', probe0);
t0 = tic;
for it = 1:budget
    if numel(queue) < bs
        queue = [queue, randperm(numel(trainIds))]; %#ok<AGROW>
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
            if useGPU, Xp1 = gpuArray(dlarray(PXb, 'SSCB')); else, Xp1 = dlarray(PXb, 'SSCB'); end
            Yp1 = forward(net, Xp1);
            pr1 = gather(extractdata(mean(Yp1(:, :, 2), 'all')));
            fprintf('TRIPWIRE probe: iter0=%.5f iter10=%.5f -> %s\n', probe0, pr1, string(pr1 >= probe0));
            if pr1 < probe0, fprintf('TRIPWIRE-FLAG: collapsing.\n'); end
        end
    else
        if lv > 2*refMed, overRel = overRel + 1; else, overRel = 0; end
        assert(overRel < 50, 'Loss > 2x reference sustained 50 iters — STOP.');
    end
    step = step + 1;
    [net.Learnables, mA, vA] = adamupdate(net.Learnables, grads, mA, vA, step, lr);
    fprintf(cf, '%d,%.6f,%.6f,%.6f,%.4e,%d,%d\n', it, lv, fo, di, np, sc < 1, overRel);
    if mod(it, 40) == 0
        fprintf('P7 iter=%d/%d loss=%.4f (focal=%.3f di=%.3f) nPre=%.4e clip=%d elapsed=%.1fmin\n', ...
            it, budget, lv, fo, di, np, sc < 1, toc(t0)/60);
    end
    if mod(it, block) == 0
        bi = it / block;
        save(fullfile(outDir, sprintf('attR_block%d_iter%d.mat', bi, it)), 'net', 'it');
        R = evalVesselEpoch(net, projRoot, valIds, useGPU, sprintf('P7 block%d', bi));
        for k = 1:numel(R)
            evalRows(end+1).block = bi; %#ok<AGROW>
            evalRows(end).id = R(k).id; evalRows(end).dFix = R(k).dFix;
            evalRows(end).dBest = R(k).dBest; evalRows(end).bestThr = R(k).bestThr;
            evalRows(end).predFrac = R(k).predFrac; evalRows(end).gtFrac = R(k).gtFrac;
            evalRows(end).fovFpFrac = R(k).fovFpFrac; evalRows(end).thinRecall = R(k).thinRecall;
        end
        fprintf('P7 BLOCK %d DONE pooled dFix=%.4f dBest=%.4f sweeps=%.1f\n', ...
            bi, mean([R.dFix]), mean([R.dBest]), it*bs/numel(trainIds));
    end
end
fclose(cf);
ef = fopen(fullfile(outDir, 'eval_P7.csv'), 'w');
fprintf(ef, 'block,id,dFix,dBest,bestThr,predFrac,gtFrac,fovFpFrac,thinRecall\n');
for i = 1:numel(evalRows)
    fprintf(ef, '%d,%d,%.5f,%.5f,%.2f,%.5f,%.5f,%.5f,%.5f\n', evalRows(i).block, ...
        evalRows(i).id, evalRows(i).dFix, evalRows(i).dBest, evalRows(i).bestThr, ...
        evalRows(i).predFrac, evalRows(i).gtFrac, evalRows(i).fovFpFrac, evalRows(i).thinRecall);
end
fclose(ef);
nB = budget / block;
byB = arrayfun(@(b) mean([evalRows([evalRows.block] == b).dFix]), 1:nB);
[~, bestB] = max(byB);
fprintf('P7 DONE bestBlock=%d pooledDfix=%.4f (P3-A control 0.3735, P6-native 0.045)\n', bestB, byB(bestB));
B2 = load(fullfile(outDir, sprintf('attR_block%d_iter%d.mat', bestB, bestB*block)), 'net');
netBest = B2.net;
for vid = [37 40]
    I = imread(fullfile(projRoot, 'dataset/DRIVE/training/images', sprintf('%d_training.tif', vid)));
    [H, W, ~] = size(I);
    I256 = imresize(I, [256 256]);
    if useGPU, Xd = gpuArray(dlarray(single(I256), 'SSC')); else, Xd = dlarray(single(I256), 'SSC'); end
    Y = forward(netBest, Xd);
    M = imresize(single(extractdata(gather(Y(:, :, 2)))) >= 0.5, [H W], 'nearest');
    imwrite(labeloverlay(I, M, 'Transparency', 0.55), ...
        fullfile(outDir, sprintf('overlay_attRBestB%d_id%d.png', bestB, vid)));
end
save(fullfile(outDir, 'summary_P7.mat'), 'evalRows', 'bestB', 'refMed', 'probe0');
diary off;
fprintf('P7 ALL_DONE outDir=%s\n', outDir);
