% runFPloc_20260922 — NEW (branch MA_FPLOC_20260922). INFERENCE ONLY.
% Where does background FP mass sit? Per FP pixel (P>=0.5 on background),
% distance to nearest GT lesion via bwdist. ep2-best vs ep3-final nets,
% fixed R.sel 50 patches. Locked read: new ep3 FP mass <=10px ->
% over-extension (boundary term next); >50px scattered -> generic collapse
% (focal gamma next). Plus 3-4 visual FP maps. No training, no test set.
projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
manDir   = 'C:/Users/SHIVANYA SALES/Downloads/MA_Patch_Manifests';
outDir   = fullfile(projRoot, 'results/MA_FPLOC_20260922');
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_FPLOC_20260922'));
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_verify_20260921'));
V = load(fullfile(manDir, 'MA_val_manifest.mat'));   v = V.valManifest;
Rt = load(fullfile(projRoot, 'results/MA_ACTFIX_20260921/rank_trained_ACTFIX_20260921.mat'));
esel = Rt.R.sel;
E2 = load(fullfile(projRoot, 'results/MA_LR3E5_20260922/best_diagnostic_checkpoint.mat'));
E3 = load(fullfile(projRoot, 'results/MA_LR3E5_20260922/final_epoch_3.mat'));
nets = {E2.bestNet, E3.net};
tags = ["ep2", "ep3"];
useGPU = (gpuDeviceCount > 0);

D = struct('tag', {}, 'fpN', {}, 'medD', {}, 'frac10', {}, 'frac50', {});
PN = zeros(50, 2);
for n = 1:2
    net = nets{n};
    allD = [];
    perN = zeros(50, 1);
    for k = 1:50
        [Xb, Tb] = cropMA512(v, esel(k));
        if useGPU, Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
        else,      Xd = dlarray(single(Xb), 'SSCB'); end
        Y = forward(net, Xd);
        P = double(extractdata(gather(Y(:, :, 2, :))));
        G = logical(Tb(:, :, 2, :));
        if ~any(G(:)), continue; end % FP-distance needs lesions present
        Q = P >= 0.5;
        FP = Q & ~G;
        perN(k) = nnz(FP);
        if nnz(FP) > 0
            Dmap = bwdist(G);
            allD = [allD; Dmap(FP)]; %#ok<AGROW>
        end
        if mod(k, 10) == 0, fprintf('[%s] PATCH %d/50\n', tags(n), k); end
    end
    D(n).tag = tags(n); D(n).fpN = numel(allD);
    D(n).medD = median(allD); D(n).frac10 = mean(allD <= 10); D(n).frac50 = mean(allD > 50);
    PN(:, n) = perN;
    fprintf('[%s] FPpx=%d medDist=%.1f frac<=10px=%.3f frac>50px=%.3f patchesWFP=%d/39\n', ...
        tags(n), numel(allD), median(allD), mean(allD <= 10), mean(allD > 50), nnz(perN > 0));
    save(fullfile(outDir, sprintf('fpdist_%s.mat', tags(n))), 'allD', 'perN');
end

% visual FP maps: 3 patches with most ep3 FP pixels (programmatic, no cherry-pick)
[~, so] = sort(PN(:, 2), 'descend');
for j = 1:min(3, nnz(PN(:, 2) > 0))
    k = so(j);
    [Xb, Tb] = cropMA512(v, esel(k));
    if useGPU, Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
    else,      Xd = dlarray(single(Xb), 'SSCB'); end
    Y = forward(nets{2}, Xd);
    P = double(extractdata(gather(Y(:, :, 2, :))));
    G = logical(Tb(:, :, 2, :));
    Ip = uint8(255 * Xb(:, :, :, 1));
    Q = P >= 0.5;
    f = figure('visible', 'off', 'position', [100 100 1500 500]);
    subplot(1, 3, 1); imshow(Ip); title(sprintf('fundus row=%d', esel(k)));
    subplot(1, 3, 2); imshow(Q); title('ep3 P>=0.5');
    subplot(1, 3, 3); imshow(G); title('GT lesions');
    saveas(f, fullfile(outDir, sprintf('fpmap_%02d_row%d.png', j, esel(k))));
    close(f);
end
save(fullfile(outDir, 'fploc_summary.mat'), 'D', 'PN');
fprintf('FPLOC_DONE\n');
