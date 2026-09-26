% runRestrictedAUC_ACTFIX_20260921 — NEW (branch MA_ACTFIX_20260921).
% Inference only. Lesion vs 10-30px ring background (FOV-guarded).
% Gates (locked): repro max-abs-diff<=1e-4 (1e-6..1e-4 = pass+note, >1e-4 STOP);
% power rule >=30/39 survive else underpowered; pass = median>=0.58 AND
% bootstrap 95% CI excludes 0.5.
projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
outDir   = fullfile(projRoot, 'results/MA_ACTFIX_20260921/restricted');
if ~isfolder(outDir), mkdir(outDir); end
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_ACTFIX_20260921'));
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_verify_20260921'));
V = load('C:/Users/SHIVANYA SALES/Downloads/MA_Patch_Manifests/MA_val_manifest.mat');
v = V.valManifest;
Rt = load(fullfile(projRoot, 'results/MA_ACTFIX_20260921/rank_trained_ACTFIX_20260921.mat'));
R = Rt.R;
C = readtable(fullfile(projRoot, 'results/MA_ACTFIX_20260921/rank_trained_ACTFIX_20260921.csv'));
S = load(fullfile(projRoot, 'results/MA_ACTFIX_20260921/MA_UNET_512_1epoch_ACTFIX_20260921.mat'), 'net');
net = S.net;
useGPU = (gpuDeviceCount > 0);

% recompute P maps + per-patch AUC (same checkpoint, inference, no sigmoid)
aucRe = nan(50, 1);
Pstore = cell(50, 1); Gstore = cell(50, 1); FOVstore = cell(50, 1);
for k = 1:50
    [Xb, Tb] = cropMA512(v, R.sel(k));
    if useGPU, Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
    else,      Xd = dlarray(single(Xb), 'SSCB'); end
    Y = forward(net, Xd);
    P = double(extractdata(gather(Y(:, :, 2, :))));
    G = logical(Tb(:, :, 2, :));
    Pstore{k} = P; Gstore{k} = G;
    Rc = double(uint8(255 * Xb(:, :, 1, 1)));
    FOVstore{k} = largestFOV(Rc > 10);
    if R.isPos(k)
        aucRe(k) = pooledAUC(P(:), double(G(:)));
    end
    if mod(k, 10) == 0, fprintf('RECOMP %d/50\n', k); end
end

% reproducibility gate vs saved CSV
d = abs(aucRe(R.isPos) - C.auroc(R.isPos));
fprintf('REPRO maxAbsDiff=%.2g\n', max(d));
assert(max(d) <= 1e-4, 'Repro mismatch >1e-4 — STOP.');
if max(d) > 1e-6, fprintf('REPRO NOTE: diff in (1e-6,1e-4], pass with note (GPU nondeterminism)\n'); end

% restricted AUC: lesion vs FOV-guarded 10-30px ring
rAUC = nan(50, 1); kept = false(50, 1); dropWhy = strings(50, 1);
for k = 1:50
    if ~R.isPos(k), dropWhy(k) = "negative"; continue; end
    G = Gstore{k}; FOV = FOVstore{k};
    ring = imdilate(G, strel('disk', 30, 0)) & ~imdilate(G, strel('disk', 10, 0)) & ~G;
    outFrac = nnz(ring & ~FOV) / max(nnz(ring), 1);
    if outFrac > 0.10
        dropWhy(k) = sprintf("ringOutsideFOV %.2f", outFrac); continue;
    end
    ring = ring & FOV;
    if nnz(ring) < 100
        dropWhy(k) = "ringTooSmall"; continue;
    end
    P = Pstore{k};
    rAUC(k) = pooledAUC([P(G); P(ring)], [ones(nnz(G),1); zeros(nnz(ring),1)]);
    kept(k) = true;
end
nKept = nnz(kept);
fprintf('KEPT %d/39 | dropped IDRiD=%d eOphtha=%d\n', nKept, ...
    nnz(~kept & R.isPos & v.Dataset(R.sel)' == "IDRiD"), ...
    nnz(~kept & R.isPos & v.Dataset(R.sel)' == "eOphtha"));
fprintf('DROP reasons: %s\n', strjoin(unique(dropWhy(~kept & R.isPos)), ', '));

if nKept < 30
    fprintf('RESTRICTED underpowered (<30) — no pass/fail recorded\n');
    rMed = median(rAUC(kept)); ci = [nan nan];
else
    rMed = median(rAUC(kept));
    rng(20260921);
    B = 2000; meds = zeros(B, 1); vv = rAUC(kept);
    for b = 1:B, meds(b) = median(vv(randi(numel(vv), numel(vv), 1))); end
    ci = prctile(meds, [2.5 97.5]);
    pass = (rMed >= 0.58) && (ci(1) > 0.5);
    fprintf('RESTRICTED median=%.4f CI95=[%.4f,%.4f] PASS=%d\n', rMed, ci(1), ci(2), pass);
end
save(fullfile(outDir, 'restricted_AUC.mat'), 'rAUC', 'kept', 'dropWhy', 'rMed', 'ci', 'nKept');
fprintf('RESTRICTED_DONE\n');

function m = largestFOV(bw)
cc = bwconncomp(bw);
if cc.NumObjects == 0, m = bw; return; end
sz = cellfun(@numel, cc.PixelIdxList);
[~, bi] = max(sz);
m = false(size(bw)); m(cc.PixelIdxList{bi}) = true;
m = imfill(m, 'holes');
end

function a = pooledAUC(scores, labels)
[~, ord] = sort(scores, 'descend');
lab = labels(ord);
P = sum(lab); N = numel(lab) - P;
if P == 0 || N == 0, a = nan; return; end
tp = cumsum(lab); fp = cumsum(~lab);
te = [0; tp/P]; fe = [0; fp/N];
a = sum(diff(fe) .* (te(1:end-1) + te(2:end)) / 2);
end
