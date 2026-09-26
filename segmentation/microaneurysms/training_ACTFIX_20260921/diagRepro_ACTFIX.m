% diagRepro_ACTFIX — inference-only diagnostic: is the 0.0074 repro gap
% GPU noise on a flat landscape, or a systematic path difference?
projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_ACTFIX_20260921'));
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_verify_20260921'));
V = load('C:/Users/SHIVANYA SALES/Downloads/MA_Patch_Manifests/MA_val_manifest.mat');
v = V.valManifest;
Rt = load(fullfile(projRoot, 'results/MA_ACTFIX_20260921/rank_trained_ACTFIX_20260921.mat'));
R = Rt.R;
C = readtable(fullfile(projRoot, 'results/MA_ACTFIX_20260921/rank_trained_ACTFIX_20260921.csv'));
S = load(fullfile(projRoot, 'results/MA_ACTFIX_20260921/MA_UNET_512_1epoch_ACTFIX_20260921.mat'), 'net');
net = S.net;
ks = find(R.isPos, 3);
for j = 1:3
    k = ks(j);
    [Xb, Tb] = cropMA512(v, R.sel(k));
    Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
    Y1 = forward(net, Xd);
    P1 = double(extractdata(gather(Y1))); P1 = P1(:, :, 2);
    Y2 = forward(net, Xd);
    P2 = double(extractdata(gather(Y2))); P2 = P2(:, :, 2);
    Ps = 1 ./ (1 + exp(-P1));
    G = logical(Tb(:, :, 2, :));
    fprintf('k=%d csv=%.4f raw1=%.4f raw2=%.4f sig=%.4f Prange=[%.4f,%.4f]\n', ...
        k, C.auroc(k), myAUC(P1(:), double(G(:))), myAUC(P2(:), double(G(:))), ...
        myAUC(Ps(:), double(G(:))), min(P1(:)), max(P1(:)));
end

function a = myAUC(s, l)
[~, o] = sort(s, 'descend');
lb = l(o); P = sum(lb); N = numel(lb) - P;
tp = cumsum(lb); fp = cumsum(~lb);
te = [0; tp/P]; fe = [0; fp/N];
a = sum(diff(fe) .* (te(1:end-1) + te(2:end)) / 2);
end
