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

%% 5. Convert class to ICDR grade
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
    otherwise
        error('Unknown classifier class: %s',className);
end

%% 6. Referable DR decision
referableDR = drGrade >= 2;

if referableDR
    referralRecommendation = "REFER FOR OPHTHALMOLOGIST REVIEW";
else
    referralRecommendation = "NO IMMEDIATE REFERRAL - ROUTINE SCREENING";
end

%% 7. Model score + calibration (temperature scaling if available)
% results/temperature_v1.mat (from calibrateTemperature) softens the
% overconfident softmax so "98%" actually means ~98% correct. Falls back
% to raw scores when the file is absent. Margin = top1-top2, entropy and
% reliability flag expose borderline cases honestly in demo + PDF.
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
    calPath = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
        'results','temperature_v1.mat');
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

%% 8. Grad-CAM
gradCAMMap = gradCAM( ...
    trainedNet, ...
    classifierInput, ...
    predictedClass, ...
    'ReductionLayer','pool5');

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
result.gradCAMFeatureLayer = "res5b_relu";
result.gradCAMReductionLayer = "pool5";
result.gradCAMNote = "Model attention map; not clinically validated lesion-level evidence.";

end

function g = gradeFromClass(className)
%% GRADEFROMCLASS - shared ICDR mapping (mirrors section 5)
switch string(className)
    case "No_DR",         g = 0;
    case "Mild",          g = 1;
    case "Moderate",      g = 2;
    case "Severe",        g = 3;
    case "Proliferate_DR",g = 4;
    otherwise,            g = NaN;
end
end
