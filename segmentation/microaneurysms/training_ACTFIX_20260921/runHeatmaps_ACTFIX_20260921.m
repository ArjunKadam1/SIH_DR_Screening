% runHeatmaps_ACTFIX_20260921 — NEW (branch MA_ACTFIX_20260921, Step A).
% Inference only. No training, no weight changes.
% 1. Heatmaps for 5 positives (AUC spread) + 2 negatives: fundus w/ GT
%    markers, P heatmap (fixed 0-1 + per-patch normalized).
% 2. Quantitative alignment: green-channel dip at GT vs 5-10px ring,
%    plus shift (+-5px) / flip variants.
% 3. Mean-subtracted repooled AUROC (39 positives).
% 4. Bootstrap 95% CI (2000 reps) on per-patch median AUC (39 positives).
% Saves PNGs + heatmap_A_report CSV/MAT into results/MA_ACTFIX_20260921/heatmaps/.

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
outDir   = fullfile(projRoot, 'results/MA_ACTFIX_20260921/heatmaps');
if ~isfolder(outDir), mkdir(outDir); end
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_ACTFIX_20260921'));
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_verify_20260921'));
V = load('C:/Users/SHIVANYA SALES/Downloads/MA_Patch_Manifests/MA_val_manifest.mat');
v = V.valManifest;
Rt = load(fullfile(projRoot, 'results/MA_ACTFIX_20260921/rank_trained_ACTFIX_20260921.mat'));
R = Rt.R;
C = readtable(fullfile(projRoot, 'results/MA_ACTFIX_20260921/rank_trained_ACTFIX_20260921.csv'));
R.aucPatch = C.auroc; R.apPatch = C.ap; R.dice05v = C.dice05;
S = load(fullfile(projRoot, 'results/MA_ACTFIX_20260921/MA_UNET_512_1epoch_ACTFIX_20260921.mat'), 'net');
net = S.net;
useGPU = (gpuDeviceCount > 0);

% ---- pick 7: positives at AUC ranks [max 75% med 25% min] + 2 negatives ----
posRows = find(R.isPos);
[~, sao] = sort(R.aucPatch(posRows));
q = [numel(sao), ceil(0.75*numel(sao)), ceil(0.5*numel(sao)), ceil(0.25*numel(sao)), 1];
pickPos = posRows(sao(q));
negRows = find(~R.isPos);
pickNeg = negRows(1:min(2, numel(negRows)));
picks = [pickPos(:); pickNeg(:)];

dipT = zeros(numel(picks), 6); % dip0, dx+5, dx-5, dy+5, dy-5, fliplr
for j = 1:numel(picks)
    k = picks(j);
    [Xb, Tb] = cropMA512(v, R.sel(k));
    if useGPU, Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
    else,      Xd = dlarray(single(Xb), 'SSCB'); end
    Y = forward(net, Xd);
    P = double(extractdata(gather(Y(:, :, 2, :)))); % raw softmax ch2
    G = logical(Tb(:, :, 2, :));
    Ip = uint8(255 * Xb(:, :, :, 1));
    Ig = double(Ip(:, :, 2));

    % heatmap figure
    f = figure('visible', 'off', 'position', [100 100 1500 500]);
    subplot(1, 3, 1); imshow(Ip); hold on;
    [gy, gx] = find(G);
    if ~isempty(gx), plot(gx, gy, 'g.', 'markersize', 3); end
    title(sprintf('fundus row=%d pos=%d', R.sel(k), R.isPos(k)));
    subplot(1, 3, 2); imagesc(P, [0 1]); axis image off; colorbar;
    title('P fixed 0-1'); colormap(gca, 'hot');
    subplot(1, 3, 3); imagesc(P); axis image off; colorbar;
    title('P per-patch norm'); colormap(gca, 'hot');
    saveas(f, fullfile(outDir, sprintf('heat_%02d_row%d.png', j, R.sel(k))));
    close(f);

    % alignment dips (positives only)
    if R.isPos(k)
        dipT(j, 1) = greenDip(Ig, G);
        dipT(j, 2) = greenDip(Ig, circshift(G, [0 5]));
        dipT(j, 3) = greenDip(Ig, circshift(G, [0 -5]));
        dipT(j, 4) = greenDip(Ig, circshift(G, [5 0]));
        dipT(j, 5) = greenDip(Ig, circshift(G, [-5 0]));
        dipT(j, 6) = greenDip(Ig, fliplr(G));
    end
    fprintf('HEAT %d/%d row=%d done\n', j, numel(picks), R.sel(k));
end
fprintf('DIP table (dip0 dx+5 dx-5 dy+5 dy-5 fliplr):\n');
disp(dipT(1:5, :));

% ---- mean-subtracted repooled AUROC (39 positives) ----
Pall = []; Gall = [];
for k = 1:50
    if ~R.isPos(k), continue; end
    [Xb, Tb] = cropMA512(v, R.sel(k));
    if useGPU, Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
    else,      Xd = dlarray(single(Xb), 'SSCB'); end
    Y = forward(net, Xd);
    P = double(extractdata(gather(Y(:, :, 2, :))));
    Ps = P - mean(P(:));
    Pall = [Pall; Ps(:)]; Gall = [Gall; double(logical(Tb(:, :, 2, :)))]; %#ok<AGROW>
end
aucMS = pooledAUC(Pall, Gall);
fprintf('MEANSUB pooledAUC=%.4f (rule: >=0.55 or within 0.05 of med %.4f)\n', aucMS, R.aucMedPatch);

% ---- bootstrap CI on per-patch median AUC ----
rng(20260921);
aucs = R.aucPatch(R.isPos);
B = 2000; meds = zeros(B, 1);
for b = 1:B
    meds(b) = median(aucs(randi(numel(aucs), numel(aucs), 1)));
end
ci = prctile(meds, [2.5 97.5]);
fprintf('BOOTSTRAP medianAUC=%.4f CI95=[%.4f, %.4f] excludes0.5=%d\n', ...
    median(aucs), ci(1), ci(2), ci(1) > 0.5);

save(fullfile(outDir, 'heatmap_A_report.mat'), 'dipT', 'aucMS', 'ci', 'picks');
fprintf('HEATMAPS_DONE\n');

function d = greenDip(Ig, G)
ring = imdilate(G, strel('disk', 10, 0)) & ~imdilate(G, strel('disk', 5, 0));
ring = ring & ~G;
if ~any(G(:)) || ~any(ring(:)), d = nan; return; end
d = mean(Ig(ring)) - mean(Ig(G));
end

function a = pooledAUC(scores, labels)
[ss, ord] = sort(scores, 'descend');
lab = labels(ord);
P = sum(lab); N = numel(lab) - P;
tp = cumsum(lab); fp = cumsum(~lab);
te = [0; tp/P]; fe = [0; fp/N];
a = sum(diff(fe) .* (te(1:end-1) + te(2:end)) / 2);
end
