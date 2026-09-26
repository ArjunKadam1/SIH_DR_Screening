function [Xb, Tb, fracs] = cropVessel256_resize(projRoot, ids)
% cropVessel256_resize — Arm A data helper (resize control).
% Loads DRIVE training images + 1st_manual GT, resizes both to 256x256
% (matches readResizeVesselPair.m / resizeVesselPair.m exactly).
% ids: vector of DRIVE training numbers, e.g. 21:36.
% Xb: 256x256x3xN single, Tb: 256x256x2xN single (ch1=bg, ch2=vessel).
% fracs: per-image vessel fraction AFTER resize (for pre-train audit).
N = numel(ids);
Xb = zeros(256, 256, 3, N, 'single');
Tb = zeros(256, 256, 2, N, 'single');
fracs = zeros(N, 1);
for j = 1:N
    id = ids(j);
    ipath = fullfile(projRoot, 'dataset/DRIVE/training/images', sprintf('%d_training.tif', id));
    mpath = fullfile(projRoot, 'dataset/DRIVE/training/1st_manual', sprintf('%d_manual1.gif', id));
    I = imread(ipath);
    rawM = imread(mpath);
    V = rawM == 255;
    I256 = imresize(I, [256 256]);
    V256 = imresize(V, [256 256], 'nearest');
    Xb(:, :, :, j) = im2single(I256);
    Tb(:, :, 1, j) = single(~V256);
    Tb(:, :, 2, j) = single(V256);
    fracs(j) = nnz(V256) / numel(V256);
end
end
