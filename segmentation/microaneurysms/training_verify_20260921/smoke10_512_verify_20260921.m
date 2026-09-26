% smoke10_512_verify_20260921
% NEW implementation (2026-09-21). 10-iteration SGD smoke test.
% Plain gradient descent, lr 1e-4, batch 4, strict 512 crops.
% Writes only to results/MA_verify_20260921/. STOP on numerical failure.

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
manDir   = 'C:/Users/SHIVANYA SALES/Downloads/MA_Patch_Manifests';
outDir   = fullfile(projRoot, 'results/MA_verify_20260921');
addpath('C:/Users/SHIVANYA SALES/Downloads'); % MA_modelGradients.m

T = load(fullfile(manDir, 'MA_train_manifest.mat')); m = T.manifest;
B = load('C:/Users/SHIVANYA SALES/Downloads/MA_UNET_baseline_architecture.mat');
net = B.net;
lr = 1e-4; bs = 4;

order = 1:40;
losses = zeros(10, 1);
for it = 1:10
    idx = order((it-1)*bs+1:it*bs);
    [Xb, Tb] = cropBatch(m, idx);
    [loss, grads] = dlfeval(@MA_modelGradients, net, ...
        dlarray(single(Xb), 'SSCB'), dlarray(single(Tb), 'SSCB'));
    lv = extractdata(loss);
    assert(isfinite(lv), sprintf('Non-finite loss at iter %d — STOP.', it));
    losses(it) = lv;
    net.Learnables = dlupdate(@(p, g) p - lr*g, net.Learnables, grads);
    fprintf('SMOKE iter=%d loss=%.4f\n', it, lv);
end
assert(all(isfinite(losses)), 'Non-finite smoke losses — STOP.');
save(fullfile(outDir, 'smoke10_20260921.mat'), 'losses', 'lr');
fprintf('SMOKE_PASS saved=%s\n', fullfile(outDir, 'smoke10_20260921.mat'));

function [Xb, Tb] = cropBatch(man, idx)
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
    Xb(:, :, :, j) = im2single(I(r.Y:r.Y+511, r.X:r.X+511, :));
    Mp = M(r.Y:r.Y+511, r.X:r.X+511);
    Tb(:, :, 1, j) = single(~Mp);
    Tb(:, :, 2, j) = single(Mp);
end
end
