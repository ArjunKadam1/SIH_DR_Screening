% checkPhase6Pipeline_TrackB_20260918 — one-batch loss/gradient smoke test.
% Mirrors the R2026a trainnet call convention: lossFcn(Y, T).
projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
imds = imageDatastore(fullfile(projRoot, 'segmentation/microaneurysms/patches/train/images'));
pxds = pixelLabelDatastore(fullfile(projRoot, 'segmentation/microaneurysms/patches/train/masks'), ...
    ["background", "MA"], [0 1]);
cds = combine(imds, pxds);
B = 4;
Xs = cell(1, B);
Ts = cell(1, B);
for i = 1:B
    d = read(cds);
    I = im2single(d{1});
    M = d{2};
    Xs{i} = I;
    Ts{i} = cat(3, single(M == 'background'), single(M == 'MA'));
end
X = dlarray(single(cat(4, Xs{:})), 'SSCB');
T = dlarray(single(cat(4, Ts{:})), 'SSCB');
fprintf('X %s | T %s\n', mat2str(size(extractdata(X))), mat2str(size(extractdata(T))));
dlnet = unet([256 256 3], 2);
Y = forward(dlnet, X); % predictions, as trainnet supplies them
L = maCombinedLoss_TrackB_20260918(Y, T); % loss-only; trainnet differentiates
fprintf('LOSS_OK value=%.4f lossClass=%s\n', extractdata(L), class(L));
