% manualTestRun_P6 — P6 best-block (attB_block12_iter480) on ACTUAL images.
% A: DRIVE test 01, 02 (blind, no vessel GT in tree) — qualitative overlays +
%    vessel% sanity (GT range on train ~7.5-8.7%) + in-FOV fraction.
% B: DRIVE train-val 37-40 (GT exists) — full-size Dice + thin-recall + maps.
% Inference: native 256 tiles stride 128, averaged (same as P6 eval).
% No training, no overwrites of prior dirs. Outputs to VES_P6_ATT/testrun_P6/.

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
addpath(fullfile(projRoot, 'segmentation/vessels/training_VES_P2B'));
outDir = fullfile(projRoot, 'results/VES_P6_ATT_20260923/testrun_P6');
if ~isfolder(outDir), mkdir(outDir); end
logPath = fullfile(outDir, 'console_testP6.log');
diary(logPath);

B = load(fullfile(projRoot, 'results/VES_P6_ATT_20260923/attB_block12_iter480.mat'), 'net');
net = B.net;
useGPU = (gpuDeviceCount > 0);
ps = 256; st = 128;
fprintf('P6-TEST net=attB_block12 (dFix 0.045 pooled val) gpu=%d\n', useGPU);

runTile = @(I) inferTiledP6(net, I, ps, st, useGPU);

% ---- A: blind DRIVE test images ----
for tid = [1 2]
    ipath = fullfile(projRoot, 'dataset/DRIVE/test/images', sprintf('%02d_test.tif', tid));
    fpath = fullfile(projRoot, 'dataset/DRIVE/test/mask', sprintf('%02d_test_mask.gif', tid));
    I = imread(ipath);
    FOV = imread(fpath) > 0;
    [H, W, ~] = size(I);
    Pfull = runTile(I);
    M = Pfull >= 0.5;
    vfrac = nnz(M)/numel(M);
    inFOV = nnz(M & FOV)/max(nnz(M), 1);
    fprintf('TEST %02d size=%dx%d vessel%%=%.2f (train-GT range 7.5-8.7) inFOV=%.3f tiles=%d\n', ...
        tid, H, W, 100*vfrac, inFOV, numel(unique([1:st:(H-ps), H-ps+1]))*numel(unique([1:st:(W-ps), W-ps+1])));
    imwrite(labeloverlay(I, M, 'Transparency', 0.55), ...
        fullfile(outDir, sprintf('DRIVEtest_%02d_overlay.png', tid)));
    figure('visible', 'off'); imagesc(Pfull, [0 1]); axis image off; colorbar; colormap hot;
    saveas(gcf, fullfile(outDir, sprintf('DRIVEtest_%02d_heatmap.png', tid))); close(gcf);
end

% ---- B: train-val with GT ----
fprintf('VAL id dFix dBest predFrac gtFrac thinRecall\n');
for vid = [37 38 39 40]
    I = imread(fullfile(projRoot, 'dataset/DRIVE/training/images', sprintf('%d_training.tif', vid)));
    GT = imread(fullfile(projRoot, 'dataset/DRIVE/training/1st_manual', sprintf('%d_manual1.gif', vid))) > 0;
    Pfull = runTile(I);
    Q = Pfull >= 0.5;
    inter = nnz(Q & GT);
    dFix = 2*inter/(nnz(Q)+nnz(GT)+eps);
    bd = 0;
    for th = 0.05:0.05:0.9
        Qq = Pfull >= th;
        dd0 = 2*nnz(Qq & GT)/(nnz(Qq)+nnz(GT)+eps);
        if dd0 > bd, bd = dd0; end
    end
    skel = bwmorph(GT, 'thin', Inf);
    D = bwdist(Q);
    tr = nnz(D(find(skel)) <= 2)/max(nnz(skel), 1);
    fprintf('VAL %d %.4f %.4f %.4f %.4f %.4f\n', vid, dFix, bd, nnz(Q)/numel(Q), nnz(GT)/numel(GT), tr);
    imwrite(labeloverlay(I, Q, 'Transparency', 0.55), ...
        fullfile(outDir, sprintf('VAL_%d_overlay.png', vid)));
end
diary off;
fprintf('P6-TEST ALL_DONE outDir=%s\n', outDir);

function Pfull = inferTiledP6(net, I, ps, st, useGPU)
[H, W, ~] = size(I);
rs = unique([1:st:(H-ps), H-ps+1]); cs = unique([1:st:(W-ps), W-ps+1]);
acc = zeros(H, W, 'single'); cnt = zeros(H, W, 'single');
for r = rs
    for c = cs
        tile = im2single(I(r:r+ps-1, c:c+ps-1, :));
        if useGPU, Xd = gpuArray(dlarray(tile, 'SSC')); else, Xd = dlarray(tile, 'SSC'); end
        Y = forward(net, Xd);
        Pv = single(extractdata(gather(Y(:, :, 2))));
        acc(r:r+ps-1, c:c+ps-1) = acc(r:r+ps-1, c:c+ps-1) + Pv;
        cnt(r:r+ps-1, c:c+ps-1) = cnt(r:r+ps-1, c:c+ps-1) + 1;
    end
end
Pfull = acc ./ max(cnt, 1);
end
