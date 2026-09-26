% trainMA_STAB_20260922 — NEW (branch MA_STAB_20260922, Phase 1 ONLY).
% Approved config: Adam 1e-4, batch 4, D3 ref loss unchanged, P=Y(:,:,2,:),
% global grad-norm clip 1.0, rng(42) order, fixed 50-patch subset, 500 iters,
% checkpoints every 50 + best-Dice tracking. No test access/aug/arch changes.
% Gate B (approved): loss>5 = explosion WARNING only (no auto-terminate).
% Hard FAIL/STOP: NaN/Inf loss or grads, or cannot reach iter 500.
% Tripwires (diagnostic, NOT success): best50 Dice>0.02 OR P-gap>0.005.

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
manDir   = 'C:/Users/SHIVANYA SALES/Downloads/MA_Patch_Manifests';
outDir   = fullfile(projRoot, 'results/MA_STAB_20260922');
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_STAB_20260922'));
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_verify_20260921'));

outNet = fullfile(outDir, 'MA_STAB_1epoch500_VERIFY.mat');
assert(~isfile(outNet), 'Output exists — STOP (never overwrite).');

seed = 42; rng(seed);
lr = 1e-4; bs = 4; cap = 500; clipNorm = 1.0;
useGPU = (gpuDeviceCount > 0);
fprintf('STAB SEED=%d LR=%.0e BATCH=%d CAP=%d CLIP=%.1f GPU=%d\n', seed, lr, bs, cap, clipNorm, useGPU);

T = load(fullfile(manDir, 'MA_train_manifest.mat')); m = T.manifest;
V = load(fullfile(manDir, 'MA_val_manifest.mat'));   v = V.valManifest;
Rt = load(fullfile(projRoot, 'results/MA_ACTFIX_20260921/rank_trained_ACTFIX_20260921.mat'));
esel = Rt.R.sel; % fixed 50-patch subset, identical to prior evals
B = load('C:/Users/SHIVANYA SALES/Downloads/MA_UNET_baseline_architecture.mat');
net = B.net;

order = randperm(height(m));
nIter = cap;
lossLog = zeros(nIter, 1); ceLog = zeros(nIter, 1); diLog = zeros(nIter, 1);
nPre = zeros(nIter, 1); nPost = zeros(nIter, 1); clipHit = false(nIter, 1);
mA = []; vA = []; step = 0;
ckpts = 50:50:cap;
CK = struct('iter', {}, 'loss', {}, 'ce', {}, 'di', {}, 'nPre', {}, 'nPost', {}, ...
    'meanPMA', {}, 'meanPBG', {}, 'gap', {}, ...
    'dFix', {}, 'iFix', {}, 'dBest', {});
bestDice = -inf; bestIter = 0;
warn5 = false;
t0 = tic;
logF = fopen(fullfile(outDir, 'train_STAB.csv'), 'w');
fprintf(logF, 'iter,loss,ce,dice,nPre,nPost,clip\n');
for it = 1:nIter
    idx = order(mod((it-1)*bs, height(m))+1:min(mod((it-1)*bs, height(m))+bs, height(m)));
    if numel(idx) < bs, idx = [idx, order(1:bs-numel(idx))]; end
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
    assert(isfinite(lv), sprintf('Non-finite loss at iter %d — FAIL/STOP.', it));
    % global grad-norm clipping (the one new mechanism)
    g2 = 0;
    for gi = 1:height(grads)
        gv = extractdata(grads.Value{gi});
        assert(all(isfinite(gv(:)), 'all'), sprintf('Non-finite grads at iter %d — FAIL/STOP.', it));
        g2 = g2 + sum(double(gv(:)).^2);
    end
    np = sqrt(g2);
    sc = min(1, clipNorm / (np + eps));
    grads = dlupdate(@(g) g*sc, grads);
    lossLog(it) = lv; ceLog(it) = gather(extractdata(ceP)); diLog(it) = gather(extractdata(diP));
    nPre(it) = np; nPost(it) = np*sc; clipHit(it) = sc < 1;
    step = step + 1;
    [net.Learnables, mA, vA] = adamupdate(net.Learnables, grads, mA, vA, step, lr);
    fprintf(logF, '%d,%.6f,%.6f,%.6f,%.4f,%.4f,%d\n', it, lv, ceLog(it), diLog(it), np, np*sc, sc < 1);
    if lv > 5, warn5 = true; end
    if mod(it, 50) == 0
        fprintf('TRAIN iter=%d loss=%.4f (ce=%.3f di=%.3f) nPre=%.2f clip=%d elapsed=%.1fmin\n', ...
            it, lv, ceLog(it), diLog(it), np, sc < 1, toc(t0)/60);
    end
    if any(ckpts == it)
        [dF, iF, dB, mMA, mBG] = eval50(net, v, esel, useGPU);
        gap = mMA - mBG;
        e = find(ckpts == it);
        CK(e).iter = it; CK(e).loss = lv; CK(e).ce = ceLog(it); CK(e).di = diLog(it);
        CK(e).nPre = np; CK(e).nPost = np*sc; CK(e).meanPMA = mMA; CK(e).meanPBG = mBG;
        CK(e).gap = gap; CK(e).dFix = dF; CK(e).iFix = iF; CK(e).dBest = dB;
        fprintf('CKPT iter=%d dFix=%.4f iFix=%.4f dBest=%.4f gap=%.5f PMA=%.4f PBG=%.4f\n', ...
            it, dF, iF, dB, gap, mMA, mBG);
        if dF > bestDice
            bestDice = dF; bestIter = it;
            bestNet = net;
            save(fullfile(outDir, 'MA_STAB_best.mat'), 'bestNet', 'bestIter', 'bestDice');
        end
    end
end
fclose(logF);
save(outNet, 'net', 'lossLog', 'ceLog', 'diLog', 'nPre', 'nPost', 'clipHit', 'CK', ...
    'bestDice', 'bestIter', 'order');
fprintf('STAB_DONE warn5=%d bestDice=%.4f@%d clipRate=%.3f\n', ...
    warn5, bestDice, bestIter, mean(clipHit));
fprintf('TRIPWIRES (diagnostic): dFix>0.02: %d | dBest>0.02: %d | gap>0.005: %d\n', ...
    any([CK.dFix] > 0.02), any([CK.dBest] > 0.02), any([CK.gap] > 0.005));

function [dF, iF, dB, mMA, mBG] = eval50(net, v, esel, useGPU)
% Returns fixed-threshold (@0.5) means AND best-threshold-sweep means.
% Best-tracking + Gate-C baseline comparison use dFix (matches prior evals);
% the locked tripwire is evaluated on dBest (D3 pass-rule convention).
ths = 0.05:0.05:0.9;
df = zeros(50, 1); iff = zeros(50, 1); db = zeros(50, 1);
sMA = 0; nMA = 0; sBG = 0; nBG = 0;
for k = 1:50
    [Xb, Tb] = cropMA512(v, esel(k));
    if useGPU, Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
    else,      Xd = dlarray(single(Xb), 'SSCB'); end
    Y = forward(net, Xd);
    P = double(extractdata(gather(Y(:, :, 2, :))));
    G = logical(Tb(:, :, 2, :));
    sMA = sMA + sum(P(G)); nMA = nMA + nnz(G);
    sBG = sBG + sum(P(~G)); nBG = nBG + nnz(~G);
    Qf = P >= 0.5;
    inter = nnz(Qf & G);
    df(k) = 2*inter / (nnz(Qf) + nnz(G) + eps);
    iff(k) = inter / (nnz(Qf | G) + eps);
    bd = 0;
    for t = 1:numel(ths)
        Q = P >= ths(t);
        dd0 = 2*nnz(Q & G) / (nnz(Q) + nnz(G) + eps);
        if dd0 > bd, bd = dd0; end
    end
    db(k) = bd;
end
dF = mean(df); iF = mean(iff); dB = mean(db);
mMA = sMA / max(nMA, 1); mBG = sBG / max(nBG, 1);
end
