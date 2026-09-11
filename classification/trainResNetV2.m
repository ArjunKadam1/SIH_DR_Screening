function trainResNetV2(varargin)
%% TRAINRESNETV2 - Anti-overfitting ResNet-18 retrain with honest V1 comparison
%
% Fixes V1's two known issues:
%   1. Overfitting (train 96.88% vs val 76.55%)
%   2. Weak Severe / Proliferative recall (F1 ~40% / ~33%)
%
% Honesty rules (do not change):
%   - Reuses V1's EXACT train/val/test split from
%     results/workspace_before_training.mat (2564/550/548) so V1-vs-V2
%     numbers are directly comparable. If that file is missing, falls back
%     to a seeded stratified split and LABELS results as fresh-split.
%   - Never overwrites results/trained_resnet18.mat (V1 baseline).
%   - Test set is touched ONLY for final evaluation, same as V1.
%   - Best model = best VALIDATION checkpoint (test picked once at end).
%
% Usage:
%   trainResNetV2                                  % Config A, GPU, 30 epochs
%   trainResNetV2('Config','A')                    % frozen head, LR 1e-4
%   trainResNetV2('Config','B')                    % full fine-tune, LR 1e-5
%   trainResNetV2('MaxEpochs',30,'MiniBatchSize',32)
%   trainResNetV2('SmokeTest',true)                 % 60-img, 1-epoch check
%   trainResNetV2('UseCache',false)                % skip disk cache (slower)
%   trainResNetV2('Config','A','ExtraData','idrid-grading') % +413 IDRiD train
%
% Name-Value options:
%   'Config'         'A' (default) or 'B'
%   'MaxEpochs'      default 30
%   'MiniBatchSize'  default 32 (GPU; smoke uses 8)
%   'InitialLearnRate' default 1e-4 for A, 1e-5 for B
%   'SmokeTest'      default false
%   'UseCache'       default true (one-time preprocess cache to disk)
%   'ValidationPatience' default 8 (epochs w/o val gain before early stop;
%                pass a large number e.g. 10000 to train full MaxEpochs)
%   'ExtraData'      'none' (default, V1-exact behavior) or 'idrid-grading'
%                adds IDRiD B Disease-Grading TRAIN images (413: No_DR 134,
%                Mild 20, Moderate 136, Severe 74, Prolif 49) to TRAIN ONLY.
%                Val/test untouched. Different camera/domain, so results are
%                labeled as augmented-train, not 1:1 V1-comparable.
%
% Outputs (per config C = A or B):
%   results/trained_resnet18_v2_C.mat  (trainedNetV2, trainInfoV2, splitSource, config)
%   results/v2_training_metrics_C.mat  (accuracy, per-class P/R/F1, referable)
%   results/v2_checkpoints_C/          (per-epoch checkpoints, best picked by val)
%   results/v2_cache_train/ + v2_cache_val/ (preprocessed 224 PNGs, reused)
%
% Requirements: Deep Learning Toolbox, Image Processing Toolbox,
%   "Deep Learning Toolbox Model for ResNet-18 Network" Add-On.
% GPU: RTX 4070 Ti SUPER confirmed (gpuDeviceCount = 1). Uses 'gpu'.

p = inputParser;
addParameter(p,'MaxEpochs',30,@isnumeric);
addParameter(p,'MiniBatchSize',32,@isnumeric);
addParameter(p,'InitialLearnRate',[],@(x) isempty(x) || isnumeric(x));
addParameter(p,'SmokeTest',false,@islogical);
addParameter(p,'Config','A',@(x) ismember(upper(string(x)),["A","B"]));
addParameter(p,'UseCache',true,@islogical);
addParameter(p,'ValidationPatience',8,@isnumeric);
addParameter(p,'ExtraData','none',@(x) ismember(lower(string(x)),["none","idrid-grading","idrid"]));
parse(p,varargin{:});
maxEpochs = p.Results.MaxEpochs;
miniBatch = p.Results.MiniBatchSize;
initLR = p.Results.InitialLearnRate;
smoke = p.Results.SmokeTest;
config = upper(char(string(p.Results.Config)));
useCache = p.Results.UseCache;
valPatience = p.Results.ValidationPatience;
extraData = lower(string(p.Results.ExtraData));

% Config defaults: B uses tiny LR for full fine-tune unless user overrode
if isempty(initLR)
    if config == "B" || config == 'B'
        initLR = 1e-5;
    else
        initLR = 1e-4;
    end
end

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot,'preprocessing'));
addpath(fullfile(projectRoot,'classification'));
resultsDir = fullfile(projectRoot,'results');

%% 1. Load V1's EXACT split (preferred) or seeded fallback
% NOTE: V1 was trained on MATLAB Online, so stored Files paths point at
% "\\MATLAB Drive\SIH_DR_Screening\dataset\...". On Desktop those paths do
% not exist, so each stored file is REMAPPED to
% <projectRoot>\dataset\<class>\<basename> (same image, same split
% membership -> still exactly comparable). Only if remap fails do we fall
% back to a fresh seeded split (labeled as such).
splitSource = "V1-exact-split (remapped to local dataset)";
v1split = fullfile(resultsDir,'workspace_before_training.mat');
datasetDir = fullfile(projectRoot,'dataset');
useFallback = true;
if isfile(v1split)
    S = load(v1split,'imdsTrain','imdsValidation','imdsTest');
    [imdsTrain, ok1] = remapSplitFiles(S.imdsTrain, datasetDir);
    [imdsVal, ok2] = remapSplitFiles(S.imdsValidation, datasetDir);
    [imdsTest, ok3] = remapSplitFiles(S.imdsTest, datasetDir);
    if ok1 && ok2 && ok3
        useFallback = false;
        fprintf('Using V1 EXACT split (remapped): train %d / val %d / test %d\n', ...
            numel(imdsTrain.Files), numel(imdsVal.Files), numel(imdsTest.Files));
    else
        warning('V1 split remap failed (local dataset missing files). Falling back.');
    end
else
    warning('V1 split file missing. Using fresh seeded split.');
end
if useFallback
    splitSource = "fresh-seeded-split (NOT 1:1 comparable with V1)";
    imds = imageDatastore( ...
        fullfile(datasetDir,{'No_DR','Mild','Moderate','Severe','Proliferate_DR'}), ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
    rng(26038); % fixed seed, documented
    [imdsTrain, tmp] = splitEachLabel(imds,0.70,'randomized');
    [imdsVal, imdsTest] = splitEachLabel(tmp,0.50,'randomized');
end

%% 1b. Optional extra training data (TRAIN ONLY, val/test untouched)
% 'idrid-grading': IDRiD B Disease-Grading TRAIN split (413 images, grades
% 0-4 -> No_DR/Mild/Moderate/Severe/Proliferate_DR). Different camera and
% resolution from APTOS, so splitSource is relabeled and results are NOT
% 1:1 V1-comparable. Useful because it adds Severe 74 + Prolif 49 +
% Moderate 136 to the scarcest classes.
if extraData == "idrid-grading" || extraData == "idrid"
    [idridFiles, idridLabels] = loadIDRiDGradingTrain(projectRoot);
    if ~isempty(idridFiles)
        imdsTrain = imageDatastore([imdsTrain.Files; idridFiles], ...
            'Labels', [imdsTrain.Labels; idridLabels]);
        splitSource = splitSource + " + IDRiD-grading-train (train-only augment)";
        fprintf('ExtraData IDRiD-grading: +%d train images (train now %d)\n', ...
            numel(idridFiles), numel(imdsTrain.Files));
    else
        warning('ExtraData idrid-grading requested but no IDRiD images found - continuing without it.');
    end
end

%% 2. Same preprocessing as V1 (Lab-L CLAHE via ReadFcn)
imdsTrain.ReadFcn = @preprocessFromFile;
imdsVal.ReadFcn = @preprocessFromFile;
imdsTest.ReadFcn = @preprocessFromFile;

%% 3. Imbalance: oversample minority classes in TRAIN ONLY (honest, documented)
% Train counts (V1): No_DR 1264, Moderate 699, Mild 259, Prolif 207, Severe 135.
% Each minority class file list is repeated up to the majority count.
% Augmentation (below) gives each replica a different random transform.
% Val/test untouched -> metrics stay comparable.
rng(26038); % fixed seed BEFORE any random draw (documented)
trainLabels = imdsTrain.Labels;
classes = categories(trainLabels);
counts = countcats(trainLabels);
target = max(counts);
repFiles = imdsTrain.Files; repLabels = trainLabels;
for c = 1:numel(classes)
    idx = find(trainLabels == classes{c});
    need = target - numel(idx);
    if need > 0
        rep = idx(randi(numel(idx),need,1)); % deterministic via rng above
        repFiles = [repFiles; imdsTrain.Files(rep)]; %#ok<AGROW>
        repLabels = [repLabels; trainLabels(rep)]; %#ok<AGROW>
    end
end
imdsTrainBal = imageDatastore(repFiles,'Labels',repLabels);
imdsTrainBal.ReadFcn = @preprocessFromFile;
fprintf('Balanced train: %d images (oversampled to %d/class)\n', ...
    numel(repFiles), target);

%% 3b. One-time preprocess cache (same pixels, just faster)
% Lab+CLAHE re-ran every image every epoch on CPU is the bottleneck
% (~3-5 min/epoch). Cache 224x224 enhanced PNGs once, reuse every epoch.
% Bypass cache for smoke test (tiny subset, not worth disk I/O).
if useCache && ~smoke
    trainCacheDir = fullfile(resultsDir,'v2_cache_train');
    valCacheDir = fullfile(resultsDir,'v2_cache_val');
    imdsTrainUsed = buildV2Cache(imdsTrainBal, trainCacheDir, [224 224]);
    imdsValUsed = buildV2Cache(imdsVal, valCacheDir, [224 224]);
    % buildV2Cache returns datastores with default imread (images already
    % enhanced) - no ReadFcn override needed.
    % Re-derive labels from cache (same multiset, folder order may differ)
    fprintf('Cache ready: train %d / val %d\n', ...
        numel(imdsTrainUsed.Files), numel(imdsValUsed.Files));
else
    imdsTrainUsed = imdsTrainBal;
    imdsValUsed = imdsVal;
    if smoke
        fprintf('Smoke test: cache bypassed.\n');
    end
end

%% 4. Stronger augmentation than V1 (anti-overfit)
augmenter = imageDataAugmenter( ...
    'RandRotation',[-20 20], ...
    'RandXTranslation',[-10 10], ...
    'RandYTranslation',[-10 10], ...
    'RandXScale',[0.9 1.1], ...
    'RandYScale',[0.9 1.1], ...
    'RandXReflection',true);
augTrain = augmentedImageDatastore([224 224 3], imdsTrainUsed, ...
    'DataAugmentation',augmenter);
augVal = augmentedImageDatastore([224 224 3], imdsValUsed);
augTest = augmentedImageDatastore([224 224 3], imdsTest);

%% 5. Network: ResNet-18, new 5-class head; freeze per config
% Config A: frozen conv1..res4 (keep res5 + head trainable) - anti-overfit.
% Config B: full fine-tune, all layers trainable, tiny LR - higher ceiling.
try
    net = resnet18('Weights','imagenet');
catch e
    error(['trainResNetV2: ResNet-18 weights missing. Install Add-On ' ...
        '"Deep Learning Toolbox Model for ResNet-18 Network". Original: %s'], e.message);
end
lgraph = layerGraph(net);
if config == "A" || config == 'A'
    % Freeze conv1..res4 (keep res5 + head trainable): LR factor 0.
    % Use replaceLayer per layer so graph connections are preserved
    % (rebuilding layerGraph from a bare array disconnects res-blocks).
    for i = 1:numel(lgraph.Layers)
        nm = lgraph.Layers(i).Name;
        if startsWith(nm,["conv1","res2","res3","res4","bn1","pool1"])
            lyr = lgraph.Layers(i);
            if isprop(lyr,'WeightLearnRateFactor')
                lyr.WeightLearnRateFactor = 0;
            end
            if isprop(lyr,'BiasLearnRateFactor')
                lyr.BiasLearnRateFactor = 0;
            end
            if isprop(lyr,'ScaleLearnRateFactor')
                lyr.ScaleLearnRateFactor = 0;
            end
            if isprop(lyr,'OffsetLearnRateFactor')
                lyr.OffsetLearnRateFactor = 0;
            end
            lgraph = replaceLayer(lgraph, nm, lyr);
        end
    end
    fprintf('Config A: frozen conv1-res4, LR %g\n', initLR);
else
    fprintf('Config B: full fine-tune, LR %g\n', initLR);
end
newFC = fullyConnectedLayer(5,'Name','fc5_v2', ...
    'WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,'fc1000',newFC);
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions', ...
    classificationLayer('Name','output5_v2'));

%% 6. Training options: GPU, 30 epochs, drops at 10/20, patience 8
% LearnRateDropPeriod 10 with 30 epochs => drops at epochs 10 and 20.
checkpointDir = fullfile(resultsDir,['v2_checkpoints_' config]);
if ~isfolder(checkpointDir)
    mkdir(checkpointDir);
end
valFreq = 30;
if smoke
    % Tiny subset for a minutes-long end-to-end check
    rng(26038);
    trIdx = randperm(numel(imdsTrainUsed.Files),min(60,numel(imdsTrainUsed.Files)));
    vaIdx = randperm(numel(imdsValUsed.Files),min(20,numel(imdsValUsed.Files)));
    augTrainSm = augmentedImageDatastore([224 224 3], ...
        subset(imdsTrainUsed,trIdx),'DataAugmentation',augmenter);
    augValSm = augmentedImageDatastore([224 224 3], subset(imdsValUsed,vaIdx));
    augTrain = augTrainSm; augVal = augValSm;
    maxEpochs = 1; miniBatch = 8; valFreq = 5;
    fprintf('SMOKE TEST (Config %s): %d train / %d val, 1 epoch.\n', ...
        config, numel(trIdx), numel(vaIdx));
end
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatch, ...
    'MaxEpochs',maxEpochs, ...
    'InitialLearnRate',initLR, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',10, ...
    'L2Regularization',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augVal, ...
    'ValidationFrequency',valFreq, ...
    'ValidationPatience',valPatience, ...
    'CheckpointPath',checkpointDir, ...
    'ExecutionEnvironment','gpu', ...
    'Verbose',true, ...
    'Plots','training-progress');

%% 7. Train (V1 file NEVER touched)
fprintf('Training V2 Config %s: epochs=%d batch=%d lr=%g gpu ...\n', ...
    config, maxEpochs, miniBatch, initLR);
try
    [trainedNetTmp, trainInfoV2] = trainNetwork(augTrain, lgraph, options);
catch e
    if contains(e.message,'GPU') || contains(e.message,'gpu') || contains(e.message,'ExecutionEnvironment')
        error(['trainResNetV2: GPU training failed (%s). Verify gpuDeviceCount returns 1 ' ...
            'and Parallel Computing Toolbox is licensed.'], e.message);
    else
        rethrow(e);
    end
end

%% 7b. Best-validation checkpoint selection (not last epoch)
% trainNetwork returns the LAST epoch net, which may already be overfit.
% Scan per-epoch checkpoints, score each on VAL ONLY (test untouched),
% keep the highest val-accuracy net as V2.
trainedNetV2 = trainedNetTmp;
valBestAcc = NaN;
if ~smoke
    ckpts = dir(fullfile(checkpointDir,'*.mat'));
    if ~isempty(ckpts)
        bestAcc = -inf;
        bestNet = trainedNetTmp;
        fprintf('Scoring %d checkpoints on val set ...\n', numel(ckpts));
        for k = 1:numel(ckpts)
            try
                C = load(fullfile(ckpts(k).folder, ckpts(k).name));
                if isfield(C,'net')
                    candNet = C.net;
                elseif isfield(C,'trainedNetV2')
                    candNet = C.trainedNetV2;
                else
                    continue;
                end
                % Checkpoints lack final batch-norm stats; aggregate them
                % over val data before scoring (otherwise classify errors).
                try
                    candNet = aggregateBatchNormalizationStatistics(candNet, augVal);
                catch
                    % non-BN nets or older releases: score directly
                end
                Yv = classify(candNet, augVal);
                accV = mean(Yv == imdsValUsed.Labels) * 100;
                if accV > bestAcc
                    bestAcc = accV;
                    bestNet = candNet;
                end
            catch ce
                warning('Checkpoint %s skipped: %s', ckpts(k).name, ce.message);
            end
        end
        if isfinite(bestAcc)
            trainedNetV2 = bestNet;
            valBestAcc = bestAcc;
            fprintf('Best checkpoint val acc: %.2f%% (last-epoch net replaced)\n', bestAcc);
        end
    else
        warning('No checkpoints found in %s; using last-epoch net.', checkpointDir);
    end
else
    fprintf('Smoke test: skipping best-checkpoint selection.\n');
end

%% 8. Evaluate on untouched test set (same metric code as V1)
[YPred, ~] = classify(trainedNetV2, augTest);
YTest = imdsTest.Labels;
acc = mean(YPred == YTest)*100;
cm = confusionmat(YTest, YPred);
prec = diag(cm)./sum(cm,1)'; rec = diag(cm)./sum(cm,2);
f1 = 2*(prec.*rec)./(prec+rec);
% Referable: grade>=2 = Moderate/Severe/Proliferate_DR
refTrue = ismember(YTest,{'Moderate','Severe','Proliferate_DR'});
refPred = ismember(YPred,{'Moderate','Severe','Proliferate_DR'});
TP=sum(refTrue&refPred); TN=sum(~refTrue&~refPred);
FP=sum(~refTrue&refPred); FN=sum(refTrue&~refPred);
sens=TP/(TP+FN)*100; spec=TN/(TN+FP)*100;
refPrec=TP/(TP+FP)*100; refF1=2*TP/(2*TP+FP+FN)*100;

%% 9. Honest V1-vs-V2 table + clinical bar
fprintf('\n===== V1 vs V2 Config %s (split: %s) =====\n', config, splitSource);
fprintf('Test acc      : V1 78.8321%%  vs  V2-%s %.4f%%\n', config, acc);
fprintf('Refer sens    : V1 84.75%%    vs  V2-%s %.2f%%\n', config, sens);
fprintf('Refer spec    : V1 97.54%%    vs  V2-%s %.2f%%\n', config, spec);
fprintf('Refer F1      : V1 90.00%%    vs  V2-%s %.2f%%\n', config, refF1);
if ~isnan(valBestAcc)
    fprintf('Best val acc  : V2-%s %.2f%% (checkpoint-selected)\n', config, valBestAcc);
end
fprintf('Per-class F1  :\n');
for c = 1:numel(classes)
    fprintf('  %-14s V2 P=%.1f R=%.1f F1=%.1f\n', classes{c}, ...
        prec(c)*100, rec(c)*100, f1(c)*100);
end
if sens >= 90
    fprintf('Clinical bar (referable sens >= 90%%): PASS\n');
else
    fprintf('Clinical bar (referable sens >= 90%%): FAIL (%.2f%%) - state honestly in pitch\n', sens);
end

%% 10. Save (never overwrite V1)
save(fullfile(resultsDir,['trained_resnet18_v2_' config '.mat']), ...
    'trainedNetV2','trainInfoV2','splitSource','config','valBestAcc','-v7.3');
save(fullfile(resultsDir,['v2_training_metrics_' config '.mat']), ...
    'acc','cm','prec','rec','f1','TP','TN','FP','FN', ...
    'sens','spec','refPrec','refF1','splitSource','config','valBestAcc');
fprintf('Saved trained_resnet18_v2_%s.mat + v2_training_metrics_%s.mat\n', config, config);
end

function [imdsNew, ok] = remapSplitFiles(imdsOld, datasetDir)
%% REMAPSPLITFILES - repoint a stored split at the local dataset copy
% Matches by <class label>/<basename>; split membership is unchanged.
ok = false; imdsNew = [];
try
    files = imdsOld.Files; labels = imdsOld.Labels;
    newFiles = strings(size(files));
    for i = 1:numel(files)
        [~,bn,ext] = fileparts(files{i});
        cand = fullfile(datasetDir, char(string(labels(i))), [bn ext]);
        if ~isfile(cand)
            fprintf('Remap miss: %s\n', cand);
            return;
        end
        newFiles(i) = string(cand);
    end
    imdsNew = imageDatastore(cellstr(newFiles),'Labels',labels);
    ok = true;
catch e
    fprintf('Remap error: %s\n', e.message);
end
end

function [filesOut, labelsOut] = loadIDRiDGradingTrain(projectRoot)
%% LOADIDRIDGRADINGTRAIN - IDRiD B grading TRAIN images as extra train data
% Maps IDRiD grades 0-4 to No_DR/Mild/Moderate/Severe/Proliferate_DR.
% Returns cellstr files + categorical labels aligned to V1 class names.
% Only the TRAIN CSV is used (413); the TEST CSV (103) is never touched.
filesOut = {}; labelsOut = categorical();
try
    imgDir = fullfile(projectRoot,'dataset','IDRiD','B. Disease Grading', ...
        '1. Original Images','a. Training Set');
    if ~isfolder(imgDir)
        % Fallback: recursive search for the training folder
        d = dir(fullfile(projectRoot,'dataset','IDRiD','B. Disease Grading','1. Original Images'));
        imgDir = "";
        for k = 1:numel(d)
            if d(k).isdir && startsWith(d(k).name,"a.")
                imgDir = fullfile(d(k).folder, d(k).name);
                break;
            end
        end
        if imgDir == ""
            return;
        end
    end
    csvPath = fullfile(projectRoot,'dataset','IDRiD','B. Disease Grading', ...
        '2. Groundtruths','a. IDRiD_Disease Grading_Training Labels.csv');
    if ~isfile(csvPath)
        return;
    end
    T = readtable(csvPath);
    % Column names vary ('Image name', 'Retinopathy grade'); resolve robustly
    imgCol = T.Properties.VariableNames{1};
    gradeCol = T.Properties.VariableNames{2};
    gradeMap = ["No_DR","Mild","Moderate","Severe","Proliferate_DR"];
    files = {}; labs = {};
    for i = 1:height(T)
        base = string(T.(imgCol)(i));
        % CSV stores 'IDRiD_001' without extension; match actual file
        cand = dir(fullfile(imgDir, base + ".*"));
        if isempty(cand)
            continue;
        end
        g = double(T.(gradeCol)(i));
        if isnan(g) || g < 0 || g > 4
            continue;
        end
        files{end+1,1} = fullfile(cand(1).folder, cand(1).name); %#ok<AGROW>
        labs{end+1,1} = char(gradeMap(g+1)); %#ok<AGROW>
    end
    if isempty(files)
        return;
    end
    filesOut = files;
    labelsOut = categorical(labs, ...
        {'No_DR','Mild','Moderate','Severe','Proliferate_DR'});
    fprintf('loadIDRiDGradingTrain: %d images mapped.\n', numel(filesOut));
catch e
    warning('loadIDRiDGradingTrain failed: %s', e.message);
    filesOut = {}; labelsOut = categorical();
end
end
