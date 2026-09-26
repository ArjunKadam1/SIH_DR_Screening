% smoke10_ACTFIX_20260921 — NEW (branch MA_ACTFIX_20260921).
% 10-iter SGD smoke with fixed loss (no sigmoid). STOP on non-finite loss.
projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
manDir   = 'C:/Users/SHIVANYA SALES/Downloads/MA_Patch_Manifests';
outDir   = fullfile(projRoot, 'results/MA_ACTFIX_20260921');
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_ACTFIX_20260921'));
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_verify_20260921'));
T = load(fullfile(manDir, 'MA_train_manifest.mat')); m = T.manifest;
B = load('C:/Users/SHIVANYA SALES/Downloads/MA_UNET_baseline_architecture.mat');
net = B.net;
lr = 1e-4; bs = 4;
losses = zeros(10, 1);
for it = 1:10
    idx = (it-1)*bs+1:it*bs;
    [Xb, Tb] = cropMA512(m, idx);
    [loss, grads] = dlfeval(@MA_modelGradients_ACTFIX_20260921, net, ...
        dlarray(single(Xb), 'SSCB'), dlarray(single(Tb), 'SSCB'));
    lv = extractdata(loss);
    assert(isfinite(lv), sprintf('Non-finite loss at iter %d — STOP.', it));
    losses(it) = lv;
    net.Learnables = dlupdate(@(p, g) p - lr*g, net.Learnables, grads);
    fprintf('SMOKE iter=%d loss=%.4f\n', it, lv);
end
save(fullfile(outDir, 'smoke10_ACTFIX_20260921.mat'), 'losses', 'lr');
fprintf('SMOKE_PASS\n');
