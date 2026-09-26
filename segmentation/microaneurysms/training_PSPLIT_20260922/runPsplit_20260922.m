% runPsplit_20260922 — NEW (branch MA_PSPLIT_20260922). INFERENCE ONLY.
% Lesion-vs-background P distributions on fixed R.sel 50 patches, for
% ep2-best net (primary) and ep3-final net (persistence check).
% Locked rule: lesion p90 < 0.3 AND background p99 < 0.1 -> magnitude
% confirmed -> smooth=1e-6 justified. Lesion p90 >= 0.5 on substantial
% patch fraction with low Dice -> localization failure -> smooth shelved.
% Else mixed -> reporter's call. No training, no weight changes.

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
manDir   = 'C:/Users/SHIVANYA SALES/Downloads/MA_Patch_Manifests';
outDir   = fullfile(projRoot, 'results/MA_PSPLIT_20260922');
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_PSPLIT_20260922'));
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_verify_20260921'));
V = load(fullfile(manDir, 'MA_val_manifest.mat'));   v = V.valManifest;
Rt = load(fullfile(projRoot, 'results/MA_ACTFIX_20260921/rank_trained_ACTFIX_20260921.mat'));
esel = Rt.R.sel;
E2 = load(fullfile(projRoot, 'results/MA_LR3E5_20260922/best_diagnostic_checkpoint.mat'));
E3 = load(fullfile(projRoot, 'results/MA_LR3E5_20260922/final_epoch_3.mat'));
nets = {E2.bestNet, E3.net};
tags = ["ep2", "ep3"];
useGPU = (gpuDeviceCount > 0);
THS = [0.1 0.3 0.5];

for n = 1:2
    net = nets{n};
    Les = []; Bg = [];
    perP90 = nan(50, 1);
    for k = 1:50
        [Xb, Tb] = cropMA512(v, esel(k));
        if useGPU, Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
        else,      Xd = dlarray(single(Xb), 'SSCB'); end
        Y = forward(net, Xd);
        P = single(extractdata(gather(Y(:, :, 2, :))));
        G = logical(Tb(:, :, 2, :));
        if any(G(:))
            lp = P(G);
            Les = [Les; lp]; %#ok<AGROW>
            perP90(k) = prctile(double(lp), 90);
        end
        bp = P(~G);
        Bg = [Bg; bp(1:10:end)]; %#ok<AGROW> % stride-10 subsample, logged
        if mod(k, 10) == 0, fprintf('[%s] PATCH %d/50\n', tags(n), k); end
    end
    Les = double(Les); Bg = double(Bg);
    fprintf('== [%s] nLes=%d nBg(sub)=%d ==\n', tags(n), numel(Les), numel(Bg));
    fprintf('lesion  p50=%.4f p90=%.4f p99=%.4f max=%.4f\n', ...
        prctile(Les,50), prctile(Les,90), prctile(Les,99), max(Les));
    fprintf('bg      p50=%.4f p90=%.4f p99=%.4f max=%.4f\n', ...
        prctile(Bg,50), prctile(Bg,90), prctile(Bg,99), max(Bg));
    for t = THS
        fprintf('thr=%.1f lesFrac=%.4f bgFrac=%.5f\n', t, mean(Les >= t), mean(Bg >= t));
    end
    fprintf('per-patch lesion-p90: med=%.4f frac>=0.5=%d/39\n', ...
        median(perP90, 'omitnan'), nnz(perP90 >= 0.5));
    f = figure('visible', 'off');
    histogram(Les, 0:0.02:1, 'normalization', 'probability'); hold on;
    histogram(Bg, 0:0.02:1, 'normalization', 'probability');
    legend('lesion', 'background'); title(sprintf('P distributions [%s]', tags(n)));
    xlabel('P'); ylabel('fraction');
    saveas(f, fullfile(outDir, sprintf('hist_%s.png', tags(n)))); close(f);
    save(fullfile(outDir, sprintf('psplit_%s.mat', tags(n))), 'Les', 'Bg', 'perP90');
end
fprintf('PSPLIT_DONE\n');
