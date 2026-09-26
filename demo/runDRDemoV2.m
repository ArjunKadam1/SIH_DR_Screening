%% runDRDemoV2 - Quality-gated demo with PDF report
% Usage: runDRDemoV2  (select image via dialog) or runDRDemoV2(imagePath)
% Options (Tier 1 second-opinion update):
%   'Crop',true  - FOV-crop before enhance/classify (default OFF; no-op on
%     borderless data, kept for foreign camera images)
%   'TTA',true   - full 5-view ensemble verdict (default OFF)
%   'SecondOpinion',false - disable the default escalation safety net
% Defaults: single-enhance pipeline + second-opinion ON (measured best).
function runDRDemoV2(imgPath, varargin)
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot,'preprocessing'));
addpath(fullfile(projectRoot,'quality'));
addpath(fullfile(projectRoot,'demo'));
addpath(fullfile(projectRoot,'classification'));
cropOn = false; ttaOn = false; soOn = true; % SO default ON (measured: sens 87.0->90.1, spec 94.8->92.3, consults 6.9%); pass 'SecondOpinion',false to opt out
for k = 1:2:numel(varargin)
    if k+1 > numel(varargin), break; end
    key = lower(string(varargin{k}));
    if key == "crop" && islogical(varargin{k+1})
        cropOn = varargin{k+1};
    elseif key == "tta" && islogical(varargin{k+1})
        ttaOn = varargin{k+1};
    elseif (key == "secondopinion" || key == "second_opinion") && islogical(varargin{k+1})
        soOn = varargin{k+1};
    end
end
% Phase-2 swap (2026-09-25): handheld R50 aptosidrid4 is the live grader.
% ROLLBACK: restore the two lines loading results/trained_resnet18.mat
% 'trainedNet' (V1 file kept on disk, untouched).
S = load('C:/Users/SHIVANYA SALES/Desktop/DR tejas/HandheldDR/results/handheld_resnet50_aptosidrid4.mat','trainedNetHandheld');
trainedNet = S.trainedNetHandheld;
fprintf('========================================\n');
fprintf('   Diabetic Retinopathy Screening Demo\n');
fprintf('========================================\n\n');
fprintf('Model loaded successfully.\n');
if nargin < 1 || isempty(imgPath)
    [f,p] = uigetfile({'*.jpg;*.jpeg;*.png','Fundus Images'},'Select Fundus Image');
    if isequal(f,0), fprintf('No image selected.\n'); return; end
    imgPath = fullfile(p,f);
end
[~,fname,ext] = fileparts(imgPath);
inputImage = imread(imgPath);
sz0 = size(inputImage);
fprintf('Image selected: %s\n', [fname ext]);
fprintf('Image size: %d x %d x %d\n', sz0(1), sz0(2), sz0(3));
if cropOn
    [inputImage, usedCrop] = cropToFOV(inputImage);
    if usedCrop
        fprintf('FOV-crop: applied (%dx%dx%d)\n', size(inputImage));
    else
        fprintf('FOV-crop: gate failed closed, using full frame\n');
    end
end
% V2 quality gate
[qRes, enhV2] = assessFundusQuality(inputImage);
[cVal, qFlag] = qualityCheck(inputImage);
fprintf('\n--- Image Quality Assessment ---\n');
fprintf('Contrast: %.2f\n', cVal);
fprintf('Quality: %s\n', qFlag);
fprintf('Quality V2: %s - %s\n', qRes.Status, qRes.Reason);
if qRes.Status == "REJECT"
    fprintf('REJECTED: recapture image before screening.\n');
    figure('Name','DR Screening - Rejected'); imshow(inputImage); title('Rejected - recapture');
    return;
end
% Enhancement parity: V2 gentle image for DISPLAY; classification input is
% the full-resolution image (single enhance inside screenFundusImage).
% Rationale (measured 2026-09-10): pre-enhancing + re-enhancing (double
% CLAHE) is identical on 224-native images but flips verdicts on full-res
% camera images (e.g. Moderate 95% -> Mild 56% on one 1424px sample).
% Single-enhance everywhere: same 548-test numbers, stable full-res.
enhancedImage = imresize(enhV2,[224 224]);
fprintf('\n--- Preprocessing ---\n');
fprintf('CLAHE enhancement applied successfully.\n');
if ttaOn
    result = screenFundusImage(inputImage, trainedNet, 'TTA', true);
elseif soOn
    % Full-resolution input so the escalation TTA views carry real content
    result = screenFundusImage(inputImage, trainedNet, 'SecondOpinion', true);
else
    result = screenFundusImage(inputImage, trainedNet);
end
fprintf('\n--- DR Classification ---\n');
fprintf('Predicted Grade: %s\n', result.predictedClass);
fprintf('Severity: %s\n', result.severity);
fprintf('ICDR Grade: %d\n', result.ICDRGrade);
fprintf('Confidence: %.2f%%\n', result.modelScorePercent);
if isfield(result,'calScorePercent')
    fprintf('Calibrated: %.2f%%\n', result.calScorePercent);
    fprintf('Margin: %.1fpp\n', result.scoreMargin);
    fprintf('Reliability: %s\n', char(string(result.reliabilityFlag)));
end
if ttaOn
    fprintf('Mode: TTA 5-view ensemble\n');
end

fprintf('\n--- Screening Result ---\n');
fprintf('Screening Decision: %s\n', result.referralRecommendation);
fprintf('Referable DR: %s\n', char(string(result.referableDR)));
if isfield(result,'ttaUsed') && result.ttaUsed
    fprintf('TTA Used: Yes (%d views)\n', result.ttaViews);
else
    fprintf('TTA Used: No\n');
end
if isfield(result,'secondOpinionConsulted') && result.secondOpinionConsulted
    fprintf('Second Opinion: Consulted (refer=%s, escalated=%s)\n', ...
        char(string(result.secondOpinionRefer)), char(string(result.escalated)));
    if result.escalated
        fprintf('Second opinion: single view was borderline non-referable; TTA ensemble escalates to REFER.\n');
    end
else
    fprintf('Second Opinion: Not consulted\n');
end

% Override with original-res overlay inputs for display
displayGradCAMOverlay(inputImage, enhancedImage, result.gradCAMMap, result);
fprintf('\n--- Explainable AI ---\n');
fprintf('Grad-CAM generated successfully.\n');
fprintf('Note: %s\n', char(string(result.gradCAMNote)));

fprintf('\n');
fprintf('========================================\n');
fprintf('          DR SCREENING REPORT\n');
fprintf('========================================\n');
fprintf('Image       : %s\n', [fname ext]);
fprintf('Image Size  : %d x %d x %d\n', sz0(1), sz0(2), sz0(3));
fprintf('Quality V1  : %s (%.2f)\n', qFlag, cVal);
fprintf('Quality V2  : %s - %s\n', qRes.Status, qRes.Reason);
fprintf('Predicted DR: %s\n', result.predictedClass);
fprintf('Severity    : %s\n', result.severity);
fprintf('ICDR Grade  : %d\n', result.ICDRGrade);
fprintf('Confidence  : %.2f%%\n', result.modelScorePercent);
if isfield(result,'calScorePercent')
    fprintf('Calibrated  : %.2f%%\n', result.calScorePercent);
    fprintf('Margin      : %.1fpp\n', result.scoreMargin);
    fprintf('Reliability : %s\n', char(string(result.reliabilityFlag)));
end
fprintf('Referable   : %s\n', char(string(result.referableDR)));
fprintf('Screening   : %s\n', result.referralRecommendation);
if isfield(result,'ttaUsed') && result.ttaUsed
    fprintf('TTA         : Yes (%d views)\n', result.ttaViews);
else
    fprintf('TTA         : No\n');
end
if isfield(result,'secondOpinionConsulted') && result.secondOpinionConsulted
    fprintf('2nd Opinion : Consulted (refer=%s, escalated=%s)\n', ...
        char(string(result.secondOpinionRefer)), char(string(result.escalated)));
else
    fprintf('2nd Opinion : Not consulted\n');
end
fprintf('Explanation : Grad-CAM generated\n');
fprintf('========================================\n');
fprintf('Note: AI-assisted screening prototype.\n');
fprintf('Clinical confirmation is recommended.\n');
fprintf('========================================\n');

outPDF = fullfile(projectRoot,'reports',sprintf('screening_report_%s.pdf',fname));
[savedFile, ~] = generateScreeningPDF(outPDF, [fname ext], qFlag, cVal, string(qRes.Status), result);
fprintf('Report saved: %s\n', savedFile);
fprintf('\nDemo completed successfully.\n');
end
