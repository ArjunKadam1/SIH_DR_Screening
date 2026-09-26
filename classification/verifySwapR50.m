% verifySwapR50 — R50 Grad-CAM layer resolution + V1 backward-compat (diary).
projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
addpath(fullfile(projRoot, 'demo'));
addpath(fullfile(projRoot, 'preprocessing'));
addpath(fullfile(projRoot, 'quality'));
addpath(fullfile(projRoot, 'classification'));
diary(fullfile(projRoot, 'results/verify_swap_r50.log'));
SH = load('C:/Users/SHIVANYA SALES/Desktop/DR tejas/HandheldDR/results/handheld_resnet50_aptosidrid4.mat', 'trainedNetHandheld');
nH = SH.trainedNetHandheld;
SV = load(fullfile(projRoot, 'results/trained_resnet18.mat'), 'trainedNet');
nV = SV.trainedNet;
I = imread(fullfile(projRoot, 'dataset/APTOS/train_images/000c1434d8d7.png'));
% 1. R50 Grad-CAM resolves (avg_pool) + deciding fields, SecondOpinion ON
rH = screenFundusImage(I, nH, 'SecondOpinion', true);
fprintf('R50: %s(%d) refer=%d esc=%d feat=%s red=%s rawMax=%.3f alpha=%.2f dec=%s\n', ...
    rH.predictedClass, rH.ICDRGrade, rH.referableDR, rH.escalated, ...
    rH.gradCAMFeatureLayer, rH.gradCAMReductionLayer, rH.gradCAMRawMax, ...
    rH.gradCAMAlphaCap, rH.gradCAMDecidingClass);
% 2. V1 rollback path: same adapter code with old net (pool5 expected)
rV = screenFundusImage(I, nV, 'SecondOpinion', true);
fprintf('V1-rollback: %s(%d) refer=%d merged=%d red=%s feat=%s\n', ...
    rV.predictedClass, rV.ICDRGrade, rV.referableDR, rV.mergedGrade, ...
    rV.gradCAMReductionLayer, rV.gradCAMFeatureLayer);
% 3. TTA path on R50 (multi-crop ensemble + map)
rT = screenFundusImage(I, nH, 'TTA', true);
fprintf('R50-TTA: %s(%d) refer=%d dec=%s\n', rT.predictedClass, ...
    rT.ICDRGrade, rT.referableDR, rT.gradCAMDecidingClass);
diary off;
fprintf('VERIFY-SWAP DONE\n');
