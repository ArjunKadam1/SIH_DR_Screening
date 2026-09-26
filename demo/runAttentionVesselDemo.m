function [vesselMask, dice] = runAttentionVesselDemo(imgPath, fovPath, gtMaskPath, outDir, varargin)
%% RUNATTENTIONVESSELDEMO - FOV-masked P6 attention vessel demo.
% Usage:
%   vesselMask = runAttentionVesselDemo()                    % dialog select
%   vesselMask = runAttentionVesselDemo(imagePath, fovPath)  % + FOV mask
%   [vesselMask, dice] = runAttentionVesselDemo(imagePath, fovPath, gtMaskPath)
%   [...] = runAttentionVesselDemo(..., outDir)              % also save
%       overlay + heatmap PNGs into outDir (created if missing).
%   [...] = runAttentionVesselDemo(..., outDir, opts)        % opts.thr (.35),
%       .minArea (20), .feather (true). Display-only when outDir empty.
%
% Net: results/VES_P6_ATT_20260923/attB_block12_iter480.mat (attention U-Net,
%   69 layers, native 256 tiles stride 128, FOV-masked output).
% FOV mask is REQUIRED for honest output (P6 predicts ~35% outside FOV raw).
% Prototype label: Dice ~0.05 val, NOT clinical. Test-27 IDRiD untouched.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, 'segmentation', 'vessels'));
addpath(fullfile(projectRoot, 'segmentation', 'vessels', 'training_VES_P2B'));

S = load(fullfile(projectRoot, 'results', 'VES_P6_ATT_20260923', ...
    'attB_block12_iter480.mat'), 'net');
net = S.net;

if nargin < 1 || isempty(imgPath)
    [f, p] = uigetfile({'*.tif;*.jpg;*.jpeg;*.png', 'Fundus Images'}, ...
        'Select Fundus Image');
    if isequal(f, 0), fprintf('No image selected.\n'); return; end
    imgPath = fullfile(p, f);
end
[~, fname, ext] = fileparts(imgPath);
I = imread(imgPath);
[H, W, ~] = size(I);
fprintf('Image: %s (%dx%dx%d)\n', [fname ext], H, W, size(I, 3));

if nargin < 2 || isempty(fovPath)
    fprintf('No FOV mask given — output unmasked (expect ~35%% outside-FOV FP).\n');
    FOV = true(H, W);
else
    FOV = imread(fovPath) > 0;
    FOV = imresize(FOV, [H W], 'nearest');
end

% --- Native tiled inference, feathered: Hann-window weighted stitching over
% reflect-padded image (seam fix), calibrated threshold (P6 dBest region
% 0.30-0.45 -> default 0.35), connected-component cleanup (drop specks).
% opts struct (all optional): .thr (0.35, used when .mode='fixed'),
% .mode ('percentile' default | 'fixed'), .targetFrac (0.09 of FOV, used when
% percentile: thr = quantile(P(FOV), 1-targetFrac), anchored to train-GT mean
% vessel density ~8.7%), .minArea (20), .feather (true).
% Rationale: P6 probs sit flat ~0.35-0.45 on unseen images, so fixed cuts
% swing dots-or-wash; percentile pins output density to anatomy instead.
% Old 4-arg calls behave as mode='percentile' (documented upgrade).
ps = 256; st = 128;
opts = struct('thr', 0.35, 'mode', 'percentile', 'targetFrac', 0.09, ...
    'minArea', 20, 'feather', true);
if nargin >= 5 && ~isempty(varargin{1})
    u = varargin{1};
    fn = fieldnames(u);
    for fi = 1:numel(fn), opts.(fn{fi}) = u.(fn{fi}); end
end
useGPU = (gpuDeviceCount > 0);
pad = 128;
Ipad = padarray(I, [pad pad], 'symmetric', 'both');
[Hp, Wp, ~] = size(Ipad);
rs = 1:st:(Hp-ps+1); cs = 1:st:(W-ps+1);
if rs(end) ~= Hp-ps+1, rs = [rs, Hp-ps+1]; end
if cs(end) ~= Wp-ps+1, cs = [cs, Wp-ps+1]; end
if opts.feather
    w1 = hann(ps, 'periodic');
    Wwin = single(w1 * w1');
else
    Wwin = ones(ps, ps, 'single');
end
acc = zeros(Hp, Wp, 'single'); cnt = zeros(Hp, Wp, 'single');
for r = rs
    for c = cs
        tile = im2single(Ipad(r:r+ps-1, c:c+ps-1, :));
        if useGPU, Xd = gpuArray(dlarray(tile, 'SSC')); else, Xd = dlarray(tile, 'SSC'); end
        Y = forward(net, Xd);
        Pv = single(extractdata(gather(Y(:, :, 2))));
        acc(r:r+ps-1, c:c+ps-1) = acc(r:r+ps-1, c:c+ps-1) + Pv .* Wwin;
        cnt(r:r+ps-1, c:c+ps-1) = cnt(r:r+ps-1, c:c+ps-1) + Wwin;
    end
end
Pfull = acc ./ max(cnt, 1e-6);
Pfull = Pfull(pad+1:pad+H, pad+1:pad+W);   % crop padding
if strcmpi(opts.mode, 'percentile')
    thrUsed = quantile(Pfull(FOV), 1 - opts.targetFrac);
else
    thrUsed = opts.thr;
end
Q = Pfull >= thrUsed;
Q = Q & FOV;
vesselMask = bwareaopen(Q, opts.minArea);  % drop isolated specks
fprintf('Infer tiles=%d feather=%d mode=%s thr=%.3f(targetFrac=%.2f) minArea=%d\n', ...
    numel(rs)*numel(cs), opts.feather, opts.mode, thrUsed, opts.targetFrac, opts.minArea);
fprintf('Vessel pixels: %d (%.2f%% of image, %.2f%% of FOV)\n', nnz(vesselMask), ...
    100*nnz(vesselMask)/numel(vesselMask), 100*nnz(vesselMask)/max(nnz(FOV), 1));

dice = NaN;
if nargin >= 3 && ~isempty(gtMaskPath)
    gt = imread(gtMaskPath) > 0;
    gt = imresize(gt, [H W], 'nearest');
    dice = 2*nnz(vesselMask & gt) / max(nnz(vesselMask)+nnz(gt), 1);
    fprintf('Dice vs expert mask (FOV-masked): %.4f\n', dice);
end

figure('Name', 'Attention Vessel (FOV-masked, prototype)', 'NumberTitle', 'off');
subplot(1, 3, 1); imshow(I);              title('Original', 'FontWeight', 'bold');
subplot(1, 3, 2); imshow(vesselMask);     title('Vessel Mask (FOV)', 'FontWeight', 'bold');
subplot(1, 3, 3); imshow(labeloverlay(I, vesselMask, 'Transparency', 0.55));
title('Overlay', 'FontWeight', 'bold');
sgtitle(sprintf('%s | vessel %.2f%% of FOV (prototype, not clinical)', ...
    [fname ext], 100*nnz(vesselMask)/max(nnz(FOV), 1)), 'FontWeight', 'bold');

% --- Heatmap (always displayed) ---
figure('Name', 'Attention Vessel Probability', 'NumberTitle', 'off');
imagesc(Pfull, [0 1]); axis image off; colorbar; colormap hot;
title(sprintf('%s | P(vessel) mean=%.4f', [fname ext], mean(Pfull(:))), ...
    'FontWeight', 'bold');

% --- Optional save (overlays + heatmaps) ---
if nargin >= 4 && ~isempty(outDir)
    if ~isfolder(outDir), mkdir(outDir); end
    ovPath = fullfile(outDir, sprintf('%s_vessel_overlay.png', fname));
    hmPath = fullfile(outDir, sprintf('%s_vessel_heatmap.png', fname));
    ov = labeloverlay(I, vesselMask, 'Transparency', 0.55);
    imwrite(ov, ovPath);
    h = figure('visible', 'off'); imagesc(Pfull, [0 1]); axis image off;
    colorbar; colormap hot;
    title(sprintf('%s | P(vessel) mean=%.4f', [fname ext], mean(Pfull(:))));
    saveas(h, hmPath); close(h);
    fprintf('Saved overlay: %s\nSaved heatmap: %s\n', ovPath, hmPath);
end
end
