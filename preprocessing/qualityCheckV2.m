function [qualityFlag, qualityReason, metrics] = qualityCheckV2(img)
% qualityCheckV2
% Multi-factor fundus image quality assessment.
%
% Checks:
% 1. Contrast
% 2. Brightness
% 3. Blur
% 4. Field of view (FOV)

% Convert to grayscale
grayImg = rgb2gray(img);
grayDouble = im2double(grayImg);

%% 1. Contrast
contrastValue = std(grayDouble(:));

%% 2. Brightness
meanBrightness = mean(grayDouble(:));

%% 3. Blur detection
laplacianFilter = fspecial('laplacian', 0.2);
laplacianResponse = imfilter(grayDouble, laplacianFilter, 'replicate');
blurScore = var(laplacianResponse(:));

%% 4. Field of View estimation

% Estimate retinal field using intensity threshold
fovMask = grayDouble > 0.08;

% Keep only the largest connected component
fovMask = bwareafilt(fovMask, 1);

% Calculate coverage
fovCoverage = sum(fovMask(:)) / numel(fovMask);

%% Thresholds
contrastThreshold = 0.045;
brightnessLow = 0.08;
brightnessHigh = 0.75;
blurThreshold = 0.0005;
%%fovThreshold = 0.20;

%% Quality decision
reasons = strings(0);

if contrastValue < contrastThreshold
    reasons(end+1) = "Low contrast";
end

if meanBrightness < brightnessLow
    reasons(end+1) = "Too dark";
elseif meanBrightness > brightnessHigh
    reasons(end+1) = "Too bright";
end

if blurScore < blurThreshold
    reasons(end+1) = "Possible blur";
end

% FOV coverage is reported as an informational metric.
% It is not currently used for the quality decision because
% the dataset background makes simple threshold-based FOV
% estimation unreliable.

if isempty(reasons)
    qualityFlag = "Good";
    qualityReason = "Image passed all quality checks";
else
    qualityFlag = "Needs Review";
    qualityReason = strjoin(reasons, ", ");
end

%% Return metrics
metrics = struct( ...
    'Contrast', contrastValue, ...
    'Brightness', meanBrightness, ...
    'BlurScore', blurScore, ...
    'FOVCoverage', fovCoverage);
end