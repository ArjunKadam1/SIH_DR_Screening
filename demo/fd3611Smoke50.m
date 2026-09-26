function results = fd3611Smoke50(nPerClass)
%% FD3611SMOKE50 - 50-image smoke on EXTERNAL FD3611 handheld-test set.
% Runs the live pipeline (handheld R50 aptosidrid4, SecondOpinion ON —
% same call as RetinaAIApp.onAnalyze) on 25 Diabetic_Retinopathy + 25
% Normal PNGs from Downloads/FD3611/FD3611 (deterministic: sorted names,
% linspace sampling, rng 7). FD3611 has no DR grades, so this is a
% referral-readout + robustness smoke: crash-free rate, referable rate per
% folder, quality mix, timing. Writes reports/fd3611_smoke_50.csv.
% Old reports untouched.
if nargin < 1 || isempty(nPerClass)
    nPerClass = 25;
end
projRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projRoot,'preprocessing'));
addpath(fullfile(projRoot,'quality'));
addpath(fullfile(projRoot,'demo'));
addpath(fullfile(projRoot,'classification'));
fdRoot = 'C:/Users/SHIVANYA SALES/Downloads/FD3611/FD3611';
S = load('C:/Users/SHIVANYA SALES/Desktop/DR tejas/HandheldDR/results/handheld_resnet50_aptosidrid4.mat','trainedNetHandheld');
trainedNet = S.trainedNetHandheld;
fprintf('Model: handheld R50 aptosidrid4 (live). FD3611 external, referral readout only.\n');
classes = ["Diabetic_Retinopathy", "Normal"];
results = struct('file',{},'trueFolder',{},'qualityStatus',{}, ...
    'predictedClass',{},'predGrade',{},'confidence',{},'calConf',{}, ...
    'referable',{},'timeSec',{},'passed',{},'error',{});
nPass = 0; tTotal = 0;
for c = 1:numel(classes)
    d = dir(fullfile(fdRoot, char(classes(c)), '*.png'));
    names = sort(string({d.name}));
    idx = unique(round(linspace(1, numel(names), min(nPerClass, numel(names)))));
    fprintf('\n[%s] testing %d images...\n', classes(c), numel(idx));
    for j = 1:numel(idx)
        fn = names(idx(j));
        R = struct('file',fn,'trueFolder',classes(c),'qualityStatus',"", ...
            'predictedClass',"",'predGrade',NaN,'confidence',NaN,'calConf',NaN, ...
            'referable',false,'timeSec',NaN,'passed',false,'error',"");
        try
            t0 = tic;
            I = imread(fullfile(fdRoot, char(classes(c)), char(fn)));
            res = screenFundusImage(I, trainedNet, 'SecondOpinion', true);
            R.timeSec = toc(t0);
            R.qualityStatus = string(res.qualityStatus);
            R.predictedClass = string(res.predictedClass);
            R.predGrade = res.ICDRGrade;
            R.confidence = res.modelScorePercent;
            R.calConf = res.calScorePercent;
            R.referable = logical(res.referableDR);
            R.passed = true; nPass = nPass + 1;
            tTotal = tTotal + R.timeSec;
            fprintf('  PASS %-22s pred=%s(%g) conf=%5.1f cal=%5.1f refer=%d %s %.1fs\n', ...
                char(fn), char(R.predictedClass), R.predGrade, R.confidence, ...
                R.calConf, R.referable, char(R.qualityStatus), R.timeSec);
        catch ME
            R.error = string(ME.message);
            fprintf('  FAIL %-22s %s\n', char(fn), char(R.error));
        end
        results(end+1) = R; %#ok<AGROW>
    end
end
n = numel(results);
fprintf('\n========================================\n');
fprintf('          FD3611 SMOKE SUMMARY (R50)\n');
fprintf('========================================\n');
fprintf('Images tested : %d\nPassed : %d\n', n, nPass);
for c = 1:numel(classes)
    sel = results([results.trueFolder] == classes(c) & [results.passed]);
    if ~isempty(sel)
        fprintf('%s: n=%d referable=%d (%.1f%%) REJECT=%d\n', classes(c), ...
            numel(sel), nnz([sel.referable]), ...
            100*nnz([sel.referable])/numel(sel), ...
            nnz([sel.qualityStatus] == "REJECT"));
    end
end
fprintf('Avg time/image: %.1fs\n', tTotal/max(nPass,1));
repDir = fullfile(projRoot,'reports');
if ~isfolder(repDir), mkdir(repDir); end
csvPath = fullfile(repDir, sprintf('fd3611_smoke_%d.csv', n));
fid = fopen(csvPath,'w');
fprintf(fid,'file,trueFolder,qualityStatus,predictedClass,predGrade,confidence,calConf,referable,timeSec,passed,error\n');
for k = 1:n
    fprintf(fid,'%s,%s,%s,%s,%g,%.2f,%.2f,%s,%.2f,%d,"%s"\n', ...
        char(results(k).file), char(results(k).trueFolder), ...
        char(results(k).qualityStatus), char(results(k).predictedClass), ...
        results(k).predGrade, results(k).confidence, results(k).calConf, ...
        char(string(results(k).referable)), results(k).timeSec, ...
        results(k).passed, char(results(k).error));
end
fclose(fid);
fprintf('CSV saved: %s\n', csvPath);
end
