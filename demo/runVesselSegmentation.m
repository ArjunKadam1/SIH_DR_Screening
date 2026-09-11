function [vesselMask, dice] = runVesselSegmentation(imgPath, gtMaskPath)
%% RUNVESSELSEGMENTATION - run trained vessel U-Net on a fundus image.
% Usage:
%   vesselMask = runVesselSegmentation()                  % pick image via dialog
%   vesselMask = runVesselSegmentation(imagePath)         % segment one image
%   [vesselMask, dice] = runVesselSegmentation(imagePath, gtMaskPath)
%       % also scores Dice vs an expert mask (e.g. DRIVE *_test_mask.gif)
%
% Net: results/vesselNet_weighted_5epoch.mat -> vesselNetWeightedTrained
%   (dlnetwork, 256x256x3 input, 2 softmax channels: 1=background, 2=vessel)

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot,'segmentation','vessels'));

S = load(fullfile(projectRoot,'results','vesselNet_weighted_5epoch.mat'), ...
    'vesselNetWeightedTrained');
net = S.vesselNetWeightedTrained;

if nargin < 1 || isempty(imgPath)
    [f,p] = uigetfile({'*.tif;*.jpg;*.jpeg;*.png','Fundus Images'}, ...
        'Select Fundus Image');
    if isequal(f,0), fprintf('No image selected.\n'); return; end
    imgPath = fullfile(p,f);
end
[~,fname,ext] = fileparts(imgPath);
I = imread(imgPath);
[H,W,~] = size(I);
fprintf('Image: %s (%dx%dx%d)\n', [fname ext], H, W, size(I,3));

% --- Preprocess exactly like training (readResizeVesselPair): 256x256 ---
I256 = imresize(I,[256 256]);

% --- Inference: softmax scores -> argmax, channel 2 = vessel ---
dlX = dlarray(single(I256),'SSC');
scores = predict(net, dlX);
scores = extractdata(gather(scores));   % 256x256x2
[~,lab] = max(scores,[],3);
vessel256 = lab == 2;

% --- Back to original resolution ---
vesselMask = imresize(vessel256,[H W],'nearest');
fprintf('Vessel pixels: %d (%.2f%% of image)\n', nnz(vesselMask), ...
    100*nnz(vesselMask)/numel(vesselMask));

% --- Optional Dice vs expert mask ---
dice = NaN;
if nargin >= 2 && ~isempty(gtMaskPath)
    gt = readVesselMask(gtMaskPath);          % logical, any size
    gt = imresize(gt,[H W],'nearest');
    dice = 2*nnz(vesselMask & gt) / max(nnz(vesselMask)+nnz(gt),1);
    fprintf('Dice vs expert mask: %.4f\n', dice);
end

% --- Display ---
figure('Name','Vessel Segmentation','NumberTitle','off');
subplot(1,3,1); imshow(I);              title('Original','FontWeight','bold');
subplot(1,3,2); imshow(vesselMask);     title('Vessel Mask','FontWeight','bold');
subplot(1,3,3); imshow(labeloverlay(I,vesselMask,'Transparency',0.55));
title('Overlay','FontWeight','bold');
sgtitle(sprintf('%s | vessel %.2f%%', [fname ext], ...
    100*nnz(vesselMask)/numel(vesselMask)),'FontWeight','bold');
end
