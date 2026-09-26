% diagD2_ACTFIX — D2 loss unit test (CPU, deterministic, no network).
% Mirrors MA_modelGradients_ACTFIX_20260921 lines 8-19 exactly:
%   Dice per-BATCH (sum 'all'), smooth=1 in numerator AND denominator;
%   focal eps=1e-6 inside both logs (log-domain clamp), gamma=2, NO alpha.
% Synthetic P on overfit patch 1 labels (IDRiD_46, lesion 357px).
% Rules: perfect lowest (else "loss bug"); expected perfect<0.5<nothing<
% inverted; saturated-wrong must be finite (else "no clamp").

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
outDir   = fullfile(projRoot, 'results/MA_ACTFIX_20260921/diag');
if ~isfolder(outDir), mkdir(outDir); end
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_verify_20260921'));
manDir = 'C:/Users/SHIVANYA SALES/Downloads/MA_Patch_Manifests';
T = load(fullfile(manDir, 'MA_train_manifest.mat')); m = T.manifest;
V = load(fullfile(projRoot, 'results/MA_ACTFIX_20260921/overfit/overfit_verdict.mat'));
p1 = V.picks(1, :);
[~, Tb] = cropMA512(p1, 1);
G = double(Tb(:, :, 2, :));
fprintf('D2 GT lesionPx=%d (manifest %d)\n', nnz(G), p1.LesionPixels);

cases = struct('name', {'perfect', 'nothing', 'half', 'inverted', 'satwrong'});
Pperf = G; Pnothing = zeros(size(G)); Phalf = 0.5*ones(size(G));
Pinv = 1 - G; Psat = 1 - G; % binary GT: inverted IS exact 0/1 saturation
Ps = {Pperf, Pnothing, Phalf, Pinv, Psat};
fprintf('%-10s %10s %10s %10s %8s\n', 'case', 'diceLoss', 'focal', 'total', 'finite');
tots = zeros(5, 1);
for i = 1:5
    P = Ps{i};
    smooth = 1;
    inter = sum(P.*G, 'all');
    dice = (2*inter + smooth) / (sum(P, 'all') + sum(G, 'all') + smooth);
    dL = 1 - dice;
    e = 1e-6;
    f = -mean(G.*(1-P).^2.*log(P+e) + (1-G).*P.^2.*log(1-P+e), 'all');
    tots(i) = dL + f;
    fprintf('%-10s %10.4f %10.4f %10.4f %8d\n', cases(i).name, dL, f, tots(i), isfinite(tots(i)));
end
fprintf('ORDER perfect=%.4f half=%.4f nothing=%.4f inverted=%.4f satwrong=%.4f\n', ...
    tots(1), tots(3), tots(2), tots(4), tots(5));
save(fullfile(outDir, 'diagD2.mat'), 'tots');
fprintf('D2_DONE\n');
