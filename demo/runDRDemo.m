%% DR Screening Demo
% Explainable AI for Diabetic Retinopathy Screening

clear;
clc;
close all;

%% Project Paths
projectRoot = '/MATLAB Drive/SIH_DR_Screening';

addpath(fullfile(projectRoot, 'preprocessing'));
addpath(fullfile(projectRoot, 'classification'));
addpath(fullfile(projectRoot, 'gradcam'));

%% Load Trained Model
load(fullfile(projectRoot, ...
    'results', 'trained_resnet18.mat'), 'trainedNet');

fprintf('========================================\n');
fprintf('   Diabetic Retinopathy Screening Demo\n');
fprintf('========================================\n\n');

fprintf('Model loaded successfully.\n');



%% Select Fundus Image

[fileName, filePath] = uigetfile( ...
    {'*.jpg;*.jpeg;*.png', 'Fundus Images'}, ...
    'Select a Fundus Image');

if isequal(fileName, 0)
    fprintf('No image selected. Demo stopped.\n');
    return;
end

inputImage = imread(fullfile(filePath, fileName));

fprintf('Image selected: %s\n', fileName);
fprintf('Image size: %d x %d x %d\n', size(inputImage));


%% Image Quality Check

[contrastValue, qualityFlag] = qualityCheck(inputImage);

fprintf('\n--- Image Quality Assessment ---\n');
fprintf('Contrast: %.2f\n', contrastValue);
fprintf('Quality: %s\n', qualityFlag);


%% Image Enhancement

enhancedImage = preprocessImage(inputImage);

fprintf('\n--- Preprocessing ---\n');
fprintf('CLAHE enhancement applied successfully.\n');


%% DR Classification

[YPred, scores] = classify(trainedNet, enhancedImage);

fprintf('\n--- DR Classification ---\n');
fprintf('Predicted Grade: %s\n', string(YPred));
fprintf('Confidence: %.2f%%\n', max(scores) * 100);

%% Referable DR Screening

referableClasses = ["Moderate", "Severe", "Proliferate_DR"];

if any(string(YPred) == referableClasses)
    screeningResult = "REFERABLE DR";
else
    screeningResult = "NON-REFERABLE DR";
end

fprintf('\n--- Screening Result ---\n');
fprintf('Screening Decision: %s\n', screeningResult);


%% Explainable AI - Grad-CAM

scoreMap = gradCAM(trainedNet, enhancedImage, YPred, ...
    'ReductionLayer', 'pool5');

fprintf('\n--- Explainable AI ---\n');
fprintf('Grad-CAM generated successfully.\n');


%% Final Visualization

figure('Name','DR Screening - Explainable AI', ...
    'NumberTitle','off');

subplot(1,4,1);
imshow(inputImage);
title('Original');

subplot(1,4,2);
imshow(enhancedImage);
title('Enhanced');

subplot(1,4,3);
imagesc(scoreMap);
axis image off;
title('Grad-CAM');
colorbar;

subplot(1,4,4);
imshow(enhancedImage);
hold on;
imagesc(scoreMap, 'AlphaData', 0.45);
axis image off;
title('Explanation');

sgtitle(sprintf('DR: %s | Confidence: %.2f%% | %s', ...
    string(YPred), max(scores)*100, screeningResult));


%% Screening Report

fprintf('\n');
fprintf('========================================\n');
fprintf('          DR SCREENING REPORT\n');
fprintf('========================================\n');
fprintf('Image       : %s\n', fileName);
fprintf('Image Quality: %s\n', qualityFlag);
fprintf('Contrast    : %.2f\n', contrastValue);
fprintf('Predicted DR: %s\n', string(YPred));
fprintf('Confidence  : %.2f%%\n', max(scores) * 100);
fprintf('Screening   : %s\n', screeningResult);
fprintf('Explanation : Grad-CAM generated\n');
fprintf('========================================\n');
fprintf('Note: AI-assisted screening prototype.\n');
fprintf('Clinical confirmation is recommended.\n');
fprintf('========================================\n');

fprintf('\nDemo completed successfully.\n');