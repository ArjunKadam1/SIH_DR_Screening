% preflight_512_verify_20260921
% NEW implementation (2026-09-21). Verifies the 512 MA pipeline inputs.
% Reads friend's files only; writes only to results/MA_verify_20260921/.
% STOP (error) on any failed check.
%
% Checks:
%  1. manifests load, expected sizes (train 4948 / val 1258)
%  2. all 202 distinct source images + 202 masks exist on this PC
%  3. mask encoding rules: IDRiD (M>0), eOphtha (M>=200); spot-check
%     LesionPixels on 20 random rows per dataset
%  4. strict 512 bounds on all rows (X+511<=W, Y+511<=H), no padding
%  5. baseline U-Net I/O + MA_modelGradients shape check on 1 batch

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
manDir   = 'C:/Users/SHIVANYA SALES/Downloads/MA_Patch_Manifests';
outDir   = fullfile(projRoot, 'results/MA_verify_20260921');
addpath('C:/Users/SHIVANYA SALES/Downloads'); % MA_modelGradients.m (read-only use)

T = load(fullfile(manDir, 'MA_train_manifest.mat')); m = T.manifest;
V = load(fullfile(manDir, 'MA_val_manifest.mat'));   v = V.valManifest;
assert(height(m) == 4948 && height(v) == 1258, 'Unexpected manifest sizes — STOP.');
fprintf('MANIFESTS train=%d val=%d\n', height(m), height(v));

allR = [m; v];
ui = unique(allR.ImagePath); um = unique(allR.MaskPath);
assert(numel(ui) == 202 && numel(um) == 202, 'Unexpected distinct file counts — STOP.');
nMissI = 0; nMissM = 0;
for i = 1:numel(ui)
    if ~isfile(locmap(ui(i))), nMissI = nMissI + 1; fprintf('MISS IMG: %s\n', ui(i)); end
end
for i = 1:numel(um)
    if ~isfile(locmap(um(i))), nMissM = nMissM + 1; fprintf('MISS MSK: %s\n', um(i)); end
end
assert(nMissI == 0 && nMissM == 0, 'Missing source files — STOP.');
fprintf('SOURCES images=202/202 masks=202/202 OK\n');

% strict bounds via headers only
nvio = 0;
for i = 1:numel(ui)
    inf = imfinfo(locmap(ui(i)));
    rows = allR(strcmp(allR.ImagePath, ui(i)), :);
    bad = any(rows.X + 511 > inf.Width | rows.Y + 511 > inf.Height);
    if bad, nvio = nvio + 1; fprintf('BOUNDS VIOLATION: %s\n', ui(i)); end
end
assert(nvio == 0, 'Strict-bounds violations — STOP (no padding allowed).');
fprintf('BOUNDS strict 512 OK on %d rows\n', height(allR));

% mask-encoding spot check: 20 random rows per dataset, LesionPixels must match
rng(42);
for ds = ["IDRiD", "eOphtha"]
    sub = allR(allR.Dataset == ds, :);
    sel = randperm(height(sub), min(20, height(sub)));
    for k = 1:numel(sel)
        r = sub(sel(k), :);
        M = imread(locmap(r.MaskPath));
        if ds == "IDRiD", Mp = M(r.Y:r.Y+511, r.X:r.X+511) > 0;
        else,              Mp = M(r.Y:r.Y+511, r.X:r.X+511) >= 200; end
        assert(nnz(Mp) == r.LesionPixels, ...
            sprintf('LesionPixels mismatch — STOP: %s X=%d Y=%d', r.ImagePath, r.X, r.Y));
    end
    fprintf('MASKRULE %s: 20/20 LesionPixels match\n', ds);
end

% net + gradient shape check on one batch of 4
B = load('C:/Users/SHIVANYA SALES/Downloads/MA_UNET_baseline_architecture.mat');
net = B.net;
assert(isequal(B.inputSize, [512 512 3]) && B.numClasses == 2, 'Unexpected net I/O — STOP.');
[Xb, Tb] = cropBatch(m, [1 2 3 4]);
X = dlarray(single(Xb), 'SSCB');
Tg = dlarray(single(Tb), 'SSCB');
[loss, grads] = dlfeval(@MA_modelGradients, net, X, Tg);
lv = extractdata(loss);
assert(isscalar(lv) && isfinite(lv), 'Loss not finite scalar — STOP.');
fprintf('GRADCHECK loss=%.4f gradTables=%d\n', lv, height(grads));

save(fullfile(outDir, 'preflight_20260921.mat'), 'm', 'v');
fprintf('PREFLIGHT_PASS saved=%s\n', fullfile(outDir, 'preflight_20260921.mat'));

function lp = locmap(p)
s = string(p);
if contains(s, 'IDRiD')
    lp = char(strrep(strrep(s, '/MATLAB Drive/SIH_DR_Screening/dataset/IDRiD/', ...
        'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening/dataset/IDRiD/'), '/', '\'));
else
    lp = char(strrep(strrep(s, '/MATLAB Drive/SIH_DR_Screening/dataset/e_optha_MA/', ...
        'C:/Users/SHIVANYA SALES/Downloads/e_optha_MA/'), '/', '\'));
end
end

function [Xb, Tb] = cropBatch(man, idx)
% Strict 512 live crops, no padding. Xb: 512x512x3xN single, Tb: 512x512x2xN single.
N = numel(idx);
Xb = zeros(512, 512, 3, N, 'single');
Tb = zeros(512, 512, 2, N, 'single');
for j = 1:N
    r = man(idx(j), :);
    s = string(r.ImagePath);
    if contains(s, 'IDRiD')
        ipath = char(strrep(strrep(s, '/MATLAB Drive/SIH_DR_Screening/dataset/IDRiD/', ...
            'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening/dataset/IDRiD/'), '/', '\'));
        sp = string(r.MaskPath);
        mpath = char(strrep(strrep(sp, '/MATLAB Drive/SIH_DR_Screening/dataset/IDRiD/', ...
            'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening/dataset/IDRiD/'), '/', '\'));
        I = imread(ipath); M = imread(mpath) > 0;
    else
        ipath = char(strrep(strrep(s, '/MATLAB Drive/SIH_DR_Screening/dataset/e_optha_MA/', ...
            'C:/Users/SHIVANYA SALES/Downloads/e_optha_MA/'), '/', '\'));
        sp = string(r.MaskPath);
        mpath = char(strrep(strrep(sp, '/MATLAB Drive/SIH_DR_Screening/dataset/e_optha_MA/', ...
            'C:/Users/SHIVANYA SALES/Downloads/e_optha_MA/'), '/', '\'));
        I = imread(ipath); M = imread(mpath) >= 200;
    end
    assert(r.X + 511 <= size(I, 2) && r.Y + 511 <= size(I, 1), 'Crop out of bounds — STOP.');
    Ip = I(r.Y:r.Y+511, r.X:r.X+511, :);
    Mp = M(r.Y:r.Y+511, r.X:r.X+511);
    Xb(:, :, :, j) = im2single(Ip);
    Tb(:, :, 1, j) = single(~Mp);
    Tb(:, :, 2, j) = single(Mp);
end
end
