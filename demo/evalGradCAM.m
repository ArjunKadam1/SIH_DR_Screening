function G = evalGradCAM(trainedNet, classifierInput, decidingClass, cropRect, frameSize, reliabilityFlag, redLayer)
% EVALGRADCAM — standalone deciding-class Grad-CAM + honest rendering struct.
% Standalone: does NOT touch screenFundusImage/RetinaAIApp (wired later).
% decidingClass : string label driving the referral card (post-escalation).
% cropRect/frameSize : geometry for gradcamNativeWarp ([] = fail-closed).
% reliabilityFlag : "high"/"borderline"/"low" (mirrors screenFundusImage).
% redLayer : optional, default 'pool5' (R18; R50 names resolved at swap time).
% Returns struct: map7, featLayer, redLayer, rawMax, heatNative, alphaCap,
% caption, decidingClass. Low reliability -> alphaCap 0.25 + caption flag.
if nargin < 7 || isempty(redLayer), redLayer = 'pool5'; end
[map7, featLayer, redLayerOut] = gradCAM(trainedNet, classifierInput, ...
    decidingClass, 'ReductionLayer', redLayer);
rawMax = max(map7(:));
heatNative = gradcamNativeWarp(map7, cropRect, frameSize, ...
    [size(classifierInput, 1), size(classifierInput, 2)]);
if string(reliabilityFlag) == "low"
    alphaCap = 0.25;
    confNote = "low confidence — muted display";
else
    alphaCap = 0.45;
    confNote = "standard display";
end
G = struct('map7', map7, 'featLayer', string(featLayer), ...
    'redLayer', string(redLayerOut), 'rawMax', rawMax, ...
    'heatNative', heatNative, 'alphaCap', alphaCap, ...
    'caption', sprintf('%s | rawmax %.3f | %s', decidingClass, rawMax, confNote), ...
    'decidingClass', string(decidingClass));
end
