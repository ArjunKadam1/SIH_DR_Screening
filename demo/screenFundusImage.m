function result = screenFundusImage(I, trainedNet, varargin)
%% SCREENFUNDUSIMAGE
% SIH 2026 V2 integrated diabetic retinopathy screening pipeline.
%
% Pipeline:
% Fundus Image
%     -> Quality Assessment
%     -> Enhancement
%     -> V1 ResNet-18
%     -> ICDR Grade 0-4
%     -> Referable DR decision
%     -> Grad-CAM
%
% Grad-CAM represents model attention and is NOT clinically
% validated lesion-level evidence.

if nargin < 2
    error('Usage: result = screenFundusImage(I, trainedNet)');
end

%% 0. Optional test-time augmentation (Tier 1 scale robustness, default OFF)
% 'TTA',true averages softmax over 5 deterministic multi-crop views built
% from the full-resolution enhanced image. Default false = today's exact
% single-224 behavior (backwards compatible).
ttaOn = false; ttaViews = 5;
soOn = false; soThreshold = 70; soViews = 10;
for k = 1:2:numel(varargin)
    if k+1 > numel(varargin), break; end
    key = lower(string(varargin{k}));
    if key == "tta" && islogical(varargin{k+1})
        ttaOn = varargin{k+1};
    elseif (key == "ttaviews" || key == "tta_views") && isnumeric(varargin{k+1})
        ttaViews = varargin{k+1};
    elseif (key == "secondopinion" || key == "second_opinion") && islogical(varargin{k+1})
        soOn = varargin{k+1};
    elseif (key == "sothreshold" || key == "so_threshold") && isnumeric(varargin{k+1})
        soThreshold = varargin{k+1};
    elseif (key == "soviews" || key == "so_views") && isnumeric(varargin{k+1})
        soViews = varargin{k+1};
    end
end

%% 1. Image quality assessment
[qualityResult, enhancedImage] = assessFundusQuality(I);

result = struct();
result.pipelineVersion = "V2";
result.quality = qualityResult;
result.originalImageSize = size(I);
result.enhancedImageSize = size(enhancedImage);

%% 2. Quality gate
result.qualityStatus = string(qualityResult.Status);
result.qualityReason = string(qualityResult.Reason);

if strcmpi(result.qualityStatus,"REJECT")
    result.classificationPerformed = false;
    result.predictedClass = "REJECTED";
    result.ICDRGrade = NaN;
    result.severity = "Ungradeable";
    result.modelScore = NaN;
    result.modelScorePercent = NaN;
    result.referableDR = false;
    result.referralRecommendation = "RECAPTURE IMAGE BEFORE SCREENING";
    result.gradCAMAvailable = false;
    result.gradCAMMap = [];
    result.gradCAMNote = "No classification performed because image quality was rejected.";
    return;
end

%% 3. Prepare image for V1 ResNet-18
classifierInput = imresize(enhancedImage,[224 224]);

if size(classifierInput,3) == 1
    classifierInput = repmat(classifierInput,1,1,3);
end

%% 4. DR classification
if ttaOn
    % Multi-view ensemble on the full-resolution enhanced image.
    % Grad-CAM below still uses the single 224 classifierInput.
    [ttaLabel, ttaScores] = ttaClassify(enhancedImage, trainedNet, ttaViews);
    predictedClass = ttaLabel;
    scores = ttaScores;
else
    [predictedClass,scores] = classify(trainedNet,classifierInput);
end
className = string(predictedClass);

%% 5. Convert class to ICDR grade (5-class V1 + 4-class handheld R50)
% Handheld merges ICDR 3+4 into SevereProlif (tiny-Severe fix). Display grade
% for merged = 3 with mergedGrade flag; referral (>=2) is unaffected since
% SevereProlif is inherently referable. V1 5-class path byte-identical.
netClasses = string(trainedNet.Layers(end).Classes);
isMerged4 = ismember("SevereProlif", netClasses);
switch className
    case "No_DR"
        drGrade = 0;
        severity = "No DR";
    case "Mild"
        drGrade = 1;
        severity = "Mild";
    case "Moderate"
        drGrade = 2;
        severity = "Moderate";
    case "Severe"
        drGrade = 3;
        severity = "Severe";
    case "Proliferate_DR"
        drGrade = 4;
        severity = "Proliferative DR";
    case "SevereProlif"
        drGrade = 3;
        severity = "Severe/Proliferative (merged 3-4)";
    otherwise
        error('Unknown classifier class: %s',className);
end
result.mergedGrade = isMerged4;

%% 6. Referable DR decision (grade rule + APTOS-anchored referScore rule)
referableDR = drGrade >= 2;
result.referralRule = "grade";

if referableDR
    referralRecommendation = "REFER FOR OPHTHALMOLOGIST REVIEW";
else
    referralRecommendation = "NO IMMEDIATE REFERRAL - ROUTINE SCREENING";
end

% APTOS-anchored referral-score rule (wired 2026-09-25, report §3b):
% referScore = P(Moderate)+P(Severe[/Prolif]) >= SIH_REFER_T (0.30, APTOS val
% 90.13/94.50). Strict SUPERSET of the grade rule: can only add referrals,
% never remove. Handheld side keeps its own 0.90 (FD3611 91.12/87.50) inside
% predictSingleImage — dual points coexist by deployment. Borderline
% grade-tier flag stays report-only (not wired).
SIH_REFER_T = 0.30;
try
    clsList = string(scores.Properties.VariableNames);
catch
    try
        clsList = string(trainedNet.Layers(end).Classes);
    catch
        clsList = strings(0, 1);
    end
end
% scores from classify/TTA are positional: rebuild order from net classes
try
    netOrder = string(trainedNet.Layers(end).Classes);
catch
    netOrder = clsList;
end
referScore = NaN;
if numel(netOrder) == numel(scores)
    iMo2 = find(netOrder == "Moderate")';
    iSe2 = find(netOrder == "SevereProlif" | netOrder == "Severe" | ...
        netOrder == "Proliferate_DR")';
    if ~isempty(iMo2) && ~isempty(iSe2)
        referScore = sum(double(scores([iMo2, iSe2])));
    end
end
result.referScore = referScore;
result.referThreshold = SIH_REFER_T;
if ~referableDR && isfinite(referScore) && referScore >= SIH_REFER_T
    referableDR = true;
    result.referralRule = "referScore";
    referralRecommendation = "REFER FOR OPHTHALMOLOGIST REVIEW (referral-score rule)";
end

%% 7. Model score + calibration (temperature scaling if available)
% Temp-file preference follows the loaded net: 4-class handheld ->
% results/temperature_handheld.mat (fit 2026-09-25, T=2.50 on 550 val);
% 5-class V1 -> results/temperature_v1.mat (absent today -> raw fallback).
% Falls back to raw scores when the file is absent. Margin = top1-top2,
% entropy and reliability flag expose borderline cases honestly in demo + PDF.
modelScore = max(scores);
modelScorePercent = double(modelScore) * 100;
sortedScores = sort(double(scores), 'descend');
if numel(sortedScores) >= 2
    scoreMargin = (sortedScores(1) - sortedScores(2)) * 100;
else
    scoreMargin = modelScorePercent;
end
scoreEntropy = -sum(double(scores) .* log(max(double(scores),1e-12)));
calTemperature = NaN; calScorePercent = modelScorePercent;
try
    if isMerged4
        calFile = 'temperature_handheld.mat';
    else
        calFile = 'temperature_v1.mat';
    end
    calPath = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
        'results', calFile);
    if isfile(calPath)
        S = load(calPath,'T');
        calTemperature = double(S.T);
        logits = log(max(double(scores),1e-12));
        calScores = exp(logits / calTemperature);
        calScores = calScores / sum(calScores);
        calScorePercent = max(calScores) * 100;
    end
catch
    % keep raw confidence on any calibration failure
end
if calScorePercent >= 85 && scoreMargin >= 20
    reliabilityFlag = "high";
elseif calScorePercent >= 70 && scoreMargin >= 10
    reliabilityFlag = "borderline";
else
    reliabilityFlag = "low";
end

%% 7b. Second opinion (escalation-only, screening-safe direction)
% If the single view says NON-referable but with low confidence, consult
% the TTA ensemble. Escalation goes one way only: non-refer -> REFER.
% A REFER verdict is never downgraded. Default OFF (opt-in).
soConsulted = false; soRefer = false; escalated = false;
if soOn && ~ttaOn && ~referableDR && modelScorePercent < soThreshold
    soConsulted = true;
    [soLabel, ~] = ttaClassify(enhancedImage, trainedNet, soViews);
    soGrade = gradeFromClass(string(soLabel));
    soRefer = soGrade >= 2;
    if soRefer
        escalated = true;
        referableDR = true;
        referralRecommendation = "REFER FOR OPHTHALMOLOGIST REVIEW (borderline single view, second-opinion escalation)";
    end
end

%% 8. Grad-CAM (deciding-class + honest rendering metadata)
% Fix: the map must explain the class driving the referral card. On
% second-opinion escalation the card says REFER while the single view said
% non-referable — mapping the single-view class would explain the loser.
% So: non-escalated -> predictedClass (unchanged behavior); escalated ->
% 'Moderate' (minimal referable class, consistent with the REFER card).
% Geometry note (verified 2026-09-25): display stretches the map onto the
% same image that was classified here, so the warp is already the exact
% inverse — no rect threading needed until a caller classifies a crop but
% displays full frame (none do today). gradCAMCropRect=[] marks that slot.
if escalated
    decidingClass = "Moderate";
else
    decidingClass = className;
end
% Reduction/feature layers resolved by net generation (R18 pool5/res5b_relu
% vs R50 avg_pool/activation_49_relu) — R50-proof, V1-compatible.
layerNames = string(arrayfun(@(l) l.Name, trainedNet.Layers, 'UniformOutput', false));
if ismember("avg_pool", layerNames)
    redLayerUse = "avg_pool";
else
    redLayerUse = "pool5";
end
[gradCAMMap, gradCAMFeat, gradCAMRed] = gradCAM( ...
    trainedNet, ...
    classifierInput, ...
    decidingClass, ...
    'ReductionLayer',redLayerUse);
gradCAMRawMax = max(gradCAMMap(:));
if string(reliabilityFlag) == "low"
    gradCAMAlphaCap = 0.25;
    gradCAMConfNote = "low confidence — muted display";
else
    gradCAMAlphaCap = 0.45;
    gradCAMConfNote = "standard display";
end

%% 9. Store complete result
result.classificationPerformed = true;
result.ttaUsed = ttaOn;
result.ttaViews = ttaViews;
result.secondOpinionConsulted = soConsulted;
result.secondOpinionRefer = soRefer;
result.escalated = escalated;
result.soThreshold = soThreshold;
result.soViews = soViews;
result.predictedClass = className;
result.ICDRGrade = drGrade;
result.severity = severity;
result.modelScores = scores;
result.modelScore = modelScore;
result.modelScorePercent = modelScorePercent;
result.calTemperature = calTemperature;
result.calScorePercent = calScorePercent;
result.scoreMargin = scoreMargin;
result.scoreEntropy = scoreEntropy;
result.reliabilityFlag = reliabilityFlag;
result.referableDR = referableDR;
result.referralRecommendation = referralRecommendation;
result.classifierInput = classifierInput;
result.gradCAMAvailable = true;
result.gradCAMMap = gradCAMMap;
result.gradCAMFeatureLayer = string(gradCAMFeat);
result.gradCAMReductionLayer = string(gradCAMRed);
result.gradCAMNote = "Model attention map; not clinically validated lesion-level evidence.";
result.gradCAMDecidingClass = string(decidingClass);
result.gradCAMRawMax = gradCAMRawMax;
result.gradCAMAlphaCap = gradCAMAlphaCap;
result.gradCAMCaption = sprintf('%s | rawmax %.3f | %s', decidingClass, ...
    gradCAMRawMax, gradCAMConfNote);
result.gradCAMCropRect = [];
result.gradCAMHeatNative = gradcamNativeWarp(gradCAMMap, [], ...
    [size(I, 1), size(I, 2)], [size(classifierInput, 1), size(classifierInput, 2)]);

end

function g = gradeFromClass(className)
%% GRADEFROMCLASS - shared ICDR mapping (mirrors section 5, incl. 4-class)
switch string(className)
    case "No_DR",         g = 0;
    case "Mild",          g = 1;
    case "Moderate",      g = 2;
    case "Severe",        g = 3;
    case "Proliferate_DR",g = 4;
    case "SevereProlif",  g = 3;
    otherwise,            g = NaN;
end
end
