% runEvalEP2_20260922 — NEW (branch MA_EVALEP2_20260922). INFERENCE ONLY.
% A. Ceiling banking on ep2-best net: per-patch Dice/IoU @0.5 + distribution,
%    lesion-level hit rate (bwconncomp; hit = >=1 predicted pixel overlap),
%    full-1258 val Dice@0.5 pooled.
% B. Ep2->ep3 split: per-patch Delta Dice (@0.5 and @best-thr) + per-patch
%    P-gap both nets. Read-out locked: uniform slide vs few-patch saturation.
% Nets loaded read-only. No training, no weight changes, no test set.

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
manDir   = 'C:/Users/SHIVANYA SALES/Downloads/MA_Patch_Manifests';
outDir   = fullfile(projRoot, 'results/MA_EVALEP2_20260922');
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_EVALEP2_20260922'));
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_verify_20260921'));
V = load(fullfile(manDir, 'MA_val_manifest.mat'));   v = V.valManifest;
Rt = load(fullfile(projRoot, 'results/MA_ACTFIX_20260921/rank_trained_ACTFIX_20260921.mat'));
esel = Rt.R.sel;
E2 = load(fullfile(projRoot, 'results/MA_LR3E5_20260922/best_diagnostic_checkpoint.mat'));
E3 = load(fullfile(projRoot, 'results/MA_LR3E5_20260922/final_epoch_3.mat'));
net2 = E2.bestNet; net3 = E3.net;
useGPU = (gpuDeviceCount > 0);
ths = 0.05:0.05:0.9;

% ---- per-patch stats, both nets ----
PN = struct('row', {}, 'isPos', {}, 'd2f', {}, 'd3f', {}, 'd2b', {}, 'd3b', {}, ...
    'gap2', {}, 'gap3', {}, 'hits', {}, 'nLes', {}, 'hitRate', {}, 'fpFrac2', {});
for k = 1:50
    [Xb, Tb] = cropMA512(v, esel(k));
    if useGPU, Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
    else,      Xd = dlarray(single(Xb), 'SSCB'); end
    P2 = double(extractdata(gather(forward(net2, Xd))));
    P2 = P2(:, :, 2);
    P3 = double(extractdata(gather(forward(net3, Xd))));
    P3 = P3(:, :, 2);
    G = logical(Tb(:, :, 2, :));
    PN(k).row = esel(k); PN(k).isPos = any(G(:));
    Q2 = P2 >= 0.5; Q3 = P3 >= 0.5;
    i2 = nnz(Q2 & G); i3 = nnz(Q3 & G);
    PN(k).d2f = 2*i2 / (nnz(Q2) + nnz(G) + eps);
    PN(k).d3f = 2*i3 / (nnz(Q3) + nnz(G) + eps);
    PN(k).gap2 = mean(P2(G)) - mean(P2(~G));
    PN(k).gap3 = mean(P3(G)) - mean(P3(~G));
    bd2 = 0; bd3 = 0;
    for t = 1:numel(ths)
        A = P2 >= ths(t); dd = 2*nnz(A & G) / (nnz(A) + nnz(G) + eps);
        if dd > bd2, bd2 = dd; end
        Bm = P3 >= ths(t); ee = 2*nnz(Bm & G) / (nnz(Bm) + nnz(G) + eps);
        if ee > bd3, bd3 = ee; end
    end
    PN(k).d2b = bd2; PN(k).d3b = bd3;
    if PN(k).isPos
        CC = bwconncomp(G);
        L = labelmatrix(CC);
        hit = false(CC.NumObjects, 1);
        ov = L(Q2);
        hit(unique(ov(ov > 0))) = true;
        PN(k).hits = nnz(hit); PN(k).nLes = CC.NumObjects;
        PN(k).hitRate = nnz(hit) / CC.NumObjects;
        PN(k).fpFrac2 = nan;
    else
        PN(k).hits = 0; PN(k).nLes = 0; PN(k).hitRate = nan;
        PN(k).fpFrac2 = mean(Q2(:));
    end
    if mod(k, 10) == 0, fprintf('PATCH %d/50\n', k); end
end
pos = [PN.isPos] == 1;
fprintf('== A. CEILING (ep2 net) ==\n');
fprintf('dFix mean=%.4f med=%.4f frac>0.1=%d/%d | dBest mean=%.4f\n', ...
    mean([PN(pos).d2f]), median([PN(pos).d2f]), nnz([PN(pos).d2f] > 0.1), nnz(pos), mean([PN(pos).d2b]));
fprintf('lesion hits %d/%d = %.3f | patches with any hit %d/%d\n', ...
    sum([PN(pos).hits]), sum([PN(pos).nLes]), ...
    sum([PN(pos).hits])/max(sum([PN(pos).nLes]),1), nnz([PN(pos).hits] > 0), nnz(pos));
fprintf('neg FPfrac mean=%.5f\n', mean([PN(~pos).fpFrac2]));
fprintf('== B. SPLIT (ep2->ep3) ==\n');
dD = [PN(pos).d3f] - [PN(pos).d2f];
fprintf('Delta dFix: mean=%+.4f med=%+.4f nNeg=%d/%d maxNeg=%.4f\n', ...
    mean(dD), median(dD), nnz(dD < 0), numel(dD), min(dD));
dB = [PN(pos).d3b] - [PN(pos).d2b];
fprintf('Delta dBest: mean=%+.4f med=%+.4f\n', mean(dB), median(dB));
fprintf('gap2 mean=%.5f gap3 mean=%.5f (positives)\n', mean([PN(pos).gap2]), mean([PN(pos).gap3]));

% ---- full-1258 val Dice@0.5 pooled (ep2 net) ----
nV = height(v); bs = 4;
tI = 0; tP = 0; tG = 0;
for s = 1:ceil(nV/bs)
    idx = (s-1)*bs+1:min(s*bs, nV);
    [Xb, Tb] = cropMA512(v, idx);
    if useGPU, Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
    else,      Xd = dlarray(single(Xb), 'SSCB'); end
    Y = forward(net2, Xd);
    P = double(extractdata(gather(Y(:, :, 2, :, :))));
    G = logical(Tb(:, :, 2, :, :));
    Q = P >= 0.5;
    tI = tI + nnz(Q & G); tP = tP + nnz(Q); tG = tG + nnz(G);
end
fprintf('FULLVAL pooled Dice@0.5 = %.4f (inter=%d pred=%d gt=%d)\n', ...
    2*tI/(tP+tG+eps), tI, tP, tG);

% ---- qualitative maps: programmatic picks, no cherry-picking ----
[~, so] = sort([PN(pos).d2f]);
pp = find(pos);
tags = ["hit_hi", "hit_med", "hit_lo", "fp_hi"];
ppos = [pp(so(end)), pp(so(ceil(numel(so)/2))), pp(so(1)), 0];
[~, sf] = sort([PN(~pos).fpFrac2]); nn = find(~pos);
ppos(4) = nn(sf(end));
for j = 1:4
    k = ppos(j);
    [Xb, Tb] = cropMA512(v, esel(k));
    if useGPU, Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
    else,      Xd = dlarray(single(Xb), 'SSCB'); end
    Y = forward(net2, Xd);
    P = double(extractdata(gather(Y(:, :, 2, :))));
    G = logical(Tb(:, :, 2, :));
    Ip = uint8(255 * Xb(:, :, :, 1));
    Q = P >= 0.5;
    ov = Ip;
    Pm = repmat(Q, [1 1 3]);
    TPm = repmat(Q & G, [1 1 3]); FPm = repmat(Q & ~G, [1 1 3]);
    FN = G & ~Q;
    green = cat(3, zeros(512,512,'uint8'), 255*ones(512,512,'uint8'), zeros(512,512,'uint8'));
    red = cat(3, 255*ones(512,512,'uint8'), zeros(512,512,'uint8'), zeros(512,512,'uint8'));
    yel = cat(3, 255*ones(512,512,'uint8'), 255*ones(512,512,'uint8'), zeros(512,512,'uint8'));
    ov(TPm) = green(TPm);   % TP green
    ov(FPm) = red(FPm);     % FP red
    fnm = repmat(FN & ~Q, [1 1 3]);
    ov(fnm) = yel(fnm);     % FN yellow
    imwrite(ov, fullfile(outDir, sprintf('map_%s_row%d.png', tags(j), esel(k))));
end
T = struct2table(PN);
writetable(T, fullfile(outDir, 'perpatch_EVALEP2.csv'));
save(fullfile(outDir, 'evalEP2.mat'), 'PN', 'tI', 'tP', 'tG');
fprintf('EVAL_DONE\n');
