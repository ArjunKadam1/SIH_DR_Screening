% manualTestRun_IDRiD53 — single-image test run (YOU run this, I only created it).
% Loads the ep2-best net (read-only) + IDRiD_53 image/GT, tiled 512/256
% forward inference, stitch, score, overlays. No training, no overwrites.
% RUN: just press Run in MATLAB, or call manualTestRun_IDRiD53 from console.

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
imgPath  = fullfile(projRoot, 'dataset/IDRiD/A. Segmentation/1. Original Images/a. Training Set/IDRiD_53.jpg');
mskPath  = fullfile(projRoot, 'dataset/IDRiD/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/1. Microaneurysms/IDRiD_53_MA.tif');
netFile  = fullfile(projRoot, 'results/MA_LR3E5_20260922/best_diagnostic_checkpoint.mat');
outDir   = fullfile(projRoot, 'results/MA_LR3E5_20260922/testrun_IDRiD53_manual');
if ~isfolder(outDir), mkdir(outDir); end

S = load(netFile); net = S.bestNet;      % ep2 net, read-only
I = imread(imgPath);
M = imread(mskPath) > 0;                 % IDRiD rule: M>0
fprintf('Image %s | MA pixels %d\n', mat2str(size(I)), nnz(M));

ps = 512; st = 256; useGPU = (gpuDeviceCount > 0);
[H, W, ~] = size(I);
rs = unique([1:st:(H-ps+1), H-ps+1]); cs = unique([1:st:(W-ps+1), W-ps+1]);
acc = zeros(H, W, 'single'); cnt = zeros(H, W, 'single');
nT = numel(rs)*numel(cs); tN = 0; t0 = tic;
for r = rs
    for c = cs
        tN = tN + 1;
        tile = im2single(I(r:r+ps-1, c:c+ps-1, :));
        if useGPU, Xd = gpuArray(dlarray(tile, 'SSC')); else, Xd = dlarray(tile, 'SSC'); end
        Y = forward(net, Xd);
        P = single(extractdata(gather(Y(:, :, 2))));   % raw softmax ch2, NO sigmoid
        acc(r:r+ps-1, c:c+ps-1) = acc(r:r+ps-1, c:c+ps-1) + P;
        cnt(r:r+ps-1, c:c+ps-1) = cnt(r:r+ps-1, c:c+ps-1) + 1;
        if mod(tN, 25) == 0, fprintf('tiles %d/%d elapsed %.1fmin\n', tN, nT, toc(t0)/60); end
    end
end
Pfull = acc ./ max(cnt, 1);

for th = [0.5]
    Q = Pfull >= th;
    inter = nnz(Q & M);
    fprintf('thr=%.2f Dice=%.4f (predPx=%d)\n', th, 2*inter/(nnz(Q)+nnz(M)+eps), nnz(Q));
end
ov = I; Pm = repmat(Pfull >= 0.5, [1 1 3]);
red = cat(3, 255*ones(H, W, 'uint8'), zeros(H, W, 'uint8'), zeros(H, W, 'uint8'));
ov(Pm) = uint8(double(I(Pm))*0.4 + double(red(Pm))*0.6);
imwrite(ov, fullfile(outDir, 'overlay_pred05.png'));
[gy, gx] = find(M);
f = figure('visible', 'off'); imshow(I); hold on; plot(gx, gy, 'g.', 'markersize', 2);
saveas(f, fullfile(outDir, 'fundus_GTdots.png')); close(f);
figure('visible', 'off'); imagesc(Pfull, [0 1]); axis image off; colorbar; colormap hot;
saveas(gcf, fullfile(outDir, 'P_heatmap.png')); close(gcf);
fprintf('Saved to %s\n', outDir);
