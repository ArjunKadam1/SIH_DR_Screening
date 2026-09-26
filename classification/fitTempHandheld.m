% fitTempHandheld — fit Guo temperature on handheld val (diary-logged).
% Writes ONLY results/temperature_handheld.mat. Deletes nothing else.
projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
addpath(fullfile(projRoot, 'classification'));
diary(fullfile(projRoot, 'results/fit_temp_handheld.log'));
S = load('C:/Users/SHIVANYA SALES/Desktop/DR tejas/HandheldDR/results/handheld_resnet50_aptosidrid4.mat', ...
    'trainedNetHandheld');
imds = imageDatastore('C:/Users/SHIVANYA SALES/Desktop/DR tejas/HandheldDR/results/cache_val_resnet50_aptosidrid4', ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
augVal = augmentedImageDatastore([224 224 3], imds);
T = calibrateTemperature(S.trainedNetHandheld, augVal, imds.Labels, ...
    'SavePath', fullfile(projRoot, 'results/temperature_handheld.mat'));
fprintf('FIT DONE T=%.2f\n', T);
diary off;
