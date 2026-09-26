% trainMAUNet_TrackB_20260918
% TRACK B ONLY — Phase 6 full U-Net training run. EXPERIMENTAL.
% This module is experimental. Never describe it as clinically validated.
%
% Data: verified Phase-5 patch datastore (256x256, bg=0 / MA=1).
% Net: unet([256 256 3], 2) depth 4 (R2026a `unet`, dlnetwork).
% Loss: maCombinedLoss_TrackB_20260918 (moderated CE [0.5 1.5] + MA Dice).
% Schedule: adam, lr 1e-3 constant, batch 32, 20 epochs, val 1x/epoch,
%   patience 100 (effectively no early stop — full run as instructed).
% Touch paths: results/MA/MA_unet_full_20260918.mat (+ .log) only.
% Never touches HandheldDR/, classification checkpoints, or Track A files.

projRoot  = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
patchRoot = fullfile(projRoot, 'segmentation/microaneurysms/patches');
outNet    = fullfile(projRoot, 'results/MA/MA_unet_full_20260918.mat');
assert(~isfile(outNet), 'Output already exists — STOP (never overwrite).');

seed = 42;
rng(seed);
classNames = ["background", "MA"];
labelIDs   = [0 1];

imdsTr = imageDatastore(fullfile(patchRoot, 'train', 'images'));
pxdsTr = pixelLabelDatastore(fullfile(patchRoot, 'train', 'masks'), classNames, labelIDs);
imdsVa = imageDatastore(fullfile(patchRoot, 'val', 'images'));
pxdsVa = pixelLabelDatastore(fullfile(patchRoot, 'val', 'masks'), classNames, labelIDs);
fprintf('TRAIN nImg=%d nMsk=%d | VAL nImg=%d nMsk=%d\n', ...
    numel(imdsTr.Files), numel(pxdsTr.Files), numel(imdsVa.Files), numel(pxdsVa.Files));
assert(numel(imdsTr.Files) == 26136 && numel(imdsVa.Files) == 7272, ...
    'Unexpected patch counts vs Phase-4 checkpoint — STOP.');

cdsTr = transform(combine(imdsTr, pxdsTr), @maAugRead);
cdsVa = transform(combine(imdsVa, pxdsVa), @maPlainRead);

dlnet = unet([256 256 3], 2);
fprintf('NET built: %s\n', class(dlnet));

miniBatch = 32;
maxEpochs = 20;
valFreq = ceil(numel(imdsTr.Files) / miniBatch); % once per epoch
opts = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatch, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', cdsVa, ...
    'ValidationFrequency', valFreq, ...
    'ValidationPatience', 100, ...
    'Plots', 'none', ...
    'Verbose', true, ...
    'VerboseFrequency', 50, ...
    'ExecutionEnvironment', 'gpu', ...
    'OutputNetwork', 'last-iteration');
fprintf('VALFREQ=%d (=1/epoch)\n', valFreq);

t0 = tic;
[net, info] = trainnet(cdsTr, dlnet, @maCombinedLoss_TrackB_20260918, opts);
elapsedH = toc(t0) / 3600;

params = struct('seed', seed, 'classNames', classNames, 'labelIDs', labelIDs, ...
    'miniBatch', miniBatch, 'maxEpochs', maxEpochs, 'learnRate', 1e-3, ...
    'valFreq', valFreq, 'elapsedHours', elapsedH);
save(outNet, 'net', 'info', 'params');
fprintf('TRAINING_DONE elapsedHours=%.2f saved=%s\n', elapsedH, outNet);

function out = maAugRead(data)
% Random x-reflection + k*90 rotation (exact for binary masks) + encode.
I = data{1};
M = data{2};
if rand < 0.5
    I = fliplr(I);
    M = fliplr(M);
end
k = randi([0 3]);
I = rot90(I, k);
M = rot90(M, k);
out = {im2single(I), cat(3, single(M == 'background'), single(M == 'MA'))};
end

function out = maPlainRead(data)
I = data{1};
M = data{2};
out = {im2single(I), cat(3, single(M == 'background'), single(M == 'MA'))};
end
