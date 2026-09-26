% trainMA_STAB3_20260922 — NEW (branch MA_STAB3_20260922).
% Controlled duration experiment: EXACTLY 3 epochs (3711 iters), then STOP.
% Identical to MA_STAB_20260922 except duration + checkpoint handling:
% baseline init, Adam 1e-4, batch 4, D3 ref loss (read-only reuse),
% P=Y(:,:,2,:), clip 1.0, rng(42) per-epoch reshuffle (epoch-1 order ==
% STAB order: built-in reproducibility check on iters 1-500), no aug,
% no test set. Hard-coded loop bounds: cannot continue past epoch 3.
% Checkpoints: final_epoch_1/2/3.mat + best_diagnostic_checkpoint.mat
% (dBest-tracked; NOT called "best model").
% Gates: NaN/Inf -> FAIL/STOP. loss>5 = warning flag only.

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
manDir   = 'C:/Users/SHIVANYA SALES/Downloads/MA_Patch_Manifests';
outDir   = fullfile(projRoot, 'results/MA_STAB3_20260922');
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_STAB3_20260922'));
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_STAB_20260922'));
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_verify_20260921'));

assert(~isfile(fullfile(outDir, 'final_epoch_3.mat')), 'Output exists — STOP (never overwrite).');

seed = 42; rng(seed);
lr = 1e-4; bs = 4; clipNorm = 1.0;
itersPerEpoch = 1237; nEpochs = 3; % 4948/4 = 1237; HARD CAP below
useGPU = (gpuDeviceCount > 0);
fprintf('STAB3 SEED=%d LR=%.0e BATCH=%d CLIP=%.1f GPU=%d EPOCHS=%d (hard cap)\n', ...
    seed, lr, bs, clipNorm, useGPU, nEpochs);

T = load(fullfile(manDir, 'MA_train_manifest.mat')); m = T.manifest;
V = load(fullfile(manDir, 'MA_val_manifest.mat'));   v = V.valManifest;
Rt = load(fullfile(projRoot, 'results/MA_ACTFIX_20260921/rank_trained_ACTFIX_20260921.mat'));
esel = Rt.R.sel;
B = load('C:/Users/SHIVANYA SALES/Downloads/MA_UNET_baseline_architecture.mat');
net = B.net;
S0 = readmatrix(fullfile(projRoot, 'results/MA_STAB_20260922/train_STAB.csv'));

mA = []; vA = []; step = 0;
EP = struct('epoch', {}, 'trainMean', {}, 'valLoss', {}, 'dFix', {}, 'iFix', {}, ...
    'dBest', {}, 'mMA', {}, 'mBG', {}, 'gap', {}, 'clipRate', {}, 'maxNpre', {});
bestD = -inf;
logF = fopen(fullfile(outDir, 'train_STAB3.csv'), 'w');
fprintf(logF, 'epoch,iter,loss,ce,dice,nPre,nPost,clip\n');
t0 = tic;
for ep = 1:nEpochs
    order = randperm(height(m));
    epLoss = zeros(itersPerEpoch, 1); epClip = false(itersPerEpoch, 1); epNpre = zeros(itersPerEpoch, 1);
    for ii = 1:itersPerEpoch
        it = (ep-1)*itersPerEpoch + ii;
        idx = order((ii-1)*bs+1:min(ii*bs, height(m)));
        [Xb, Tb] = cropMA512(m, idx);
        if useGPU
            Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
            Td = gpuArray(dlarray(single(Tb), 'SSCB'));
        else
            Xd = dlarray(single(Xb), 'SSCB');
            Td = dlarray(single(Tb), 'SSCB');
        end
        [loss, grads, ceP, diP] = dlfeval(@refLoss_STAB_20260922, net, Xd, Td);
        lv = gather(extractdata(loss));
        assert(isfinite(lv), sprintf('Non-finite loss ep%d iter %d — FAIL/STOP.', ep, it));
        g2 = 0;
        for gi = 1:height(grads)
            gv = extractdata(grads.Value{gi});
            assert(all(isfinite(gv(:)), 'all'), sprintf('Non-finite grads ep%d iter %d — FAIL/STOP.', ep, it));
            g2 = g2 + sum(double(gv(:)).^2);
        end
        np = sqrt(g2);
        sc = min(1, clipNorm / (np + eps));
        grads = dlupdate(@(g) g*sc, grads);
        step = step + 1;
        [net.Learnables, mA, vA] = adamupdate(net.Learnables, grads, mA, vA, step, lr);
        epLoss(ii) = lv; epClip(ii) = sc < 1; epNpre(ii) = np;
        fprintf(logF, '%d,%d,%.6f,%.6f,%.6f,%.4f,%.4f,%d\n', ep, it, lv, ...
            gather(extractdata(ceP)), gather(extractdata(diP)), np, np*sc, sc < 1);
        if mod(ii, 250) == 0 || ii == itersPerEpoch
            fprintf('EP%d iter=%d/%d loss=%.4f nPre=%.2f clip=%d elapsed=%.1fmin\n', ...
                ep, it, nEpochs*itersPerEpoch, lv, np, sc < 1, toc(t0)/60);
        end
    end
    if ep == 1
        d = abs(epLoss(1:500) - S0(1:500, 2)); % csv cols: iter,loss,ce,...
        fprintf('REPRO vs STAB iters1-500: meanAbs=%.2g maxAbs=%.2g\n', mean(d), max(d));
        assert(mean(d) <= 0.05, 'Repro divergence >0.05 — STOP.');
    end
    % epoch-end evals: fixed-50 + full val
    [dF, iF, dB, mMA, mBG] = eval50_STAB3(net, v, esel, useGPU);
    nV = ceil(height(v) / bs);
    vL = zeros(nV, 1);
    for jt = 1:nV
        jdx = (jt-1)*bs+1:min(jt*bs, height(v));
        [Xb, Tb] = cropMA512(v, jdx);
        if useGPU
            Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
            Td = gpuArray(dlarray(single(Tb), 'SSCB'));
        else
            Xd = dlarray(single(Xb), 'SSCB');
            Td = dlarray(single(Tb), 'SSCB');
        end
        Y = forward(net, Xd);
        P = extractdata(gather(Y(:, :, 2, :)));
        G = Tb(:, :, 2, :);
        smooth = 1;
        dice = (2*sum(P .* G, [1 2]) + smooth) ./ (sum(P, [1 2]) + sum(G, [1 2]) + smooth);
        vL(jt) = gather(1 - mean(dice));
    end
    EP(ep).epoch = ep; EP(ep).trainMean = mean(epLoss); EP(ep).valLoss = mean(vL);
    EP(ep).dFix = dF; EP(ep).iFix = iF; EP(ep).dBest = dB;
    EP(ep).mMA = mMA; EP(ep).mBG = mBG; EP(ep).gap = mMA - mBG;
    EP(ep).clipRate = mean(epClip); EP(ep).maxNpre = max(epNpre);
    fprintf('EPOCH%d DONE trainMean=%.4f valLoss=%.4f dFix=%.4f iFix=%.4f dBest=%.4f gap=%.5f clipRate=%.3f maxNpre=%.1f\n', ...
        ep, mean(epLoss), mean(vL), dF, iF, dB, mMA - mBG, mean(epClip), max(epNpre));
    save(fullfile(outDir, sprintf('final_epoch_%d.mat', ep)), 'net', 'epLoss', 'epClip', 'epNpre', 'EP');
    if dB > bestD
        bestD = dB;
        bestNet = net; bestEp = ep;
        save(fullfile(outDir, 'best_diagnostic_checkpoint.mat'), 'bestNet', 'bestEp', 'bestD');
    end
end
fclose(logF);
save(fullfile(outDir, 'summary_STAB3.mat'), 'EP', 'bestD', 'bestEp');
fprintf('STAB3_DONE bestD=%.4f@ep%d — STOP (no further epochs)\n', bestD, bestEp);
