function results = rawSmoke50(nPerGrade)
%% RAWSMOKE50 - accuracy smoke test on RAW APTOS images (single enhancement).
% Why raw: screenFundusImage enhances internally; feeding it preprocessed
% 224 cache PNGs double-enhances (mean diff ~9/255) which R50 does not
% tolerate (diag_rawvscache.log). Real demo flow uploads raw camera images.
% Samples nPerGrade images per ICDR grade (3+4 merged truth=3) from
% dataset/APTOS/train.csv, runs live pipeline (SecondOpinion ON).
% Writes: reports/raw_smoke_<n>.csv. V1 file untouched.
if nargin < 1 || isempty(nPerGrade)
    nPerGrade = 10;
end
projRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projRoot,'preprocessing'));
addpath(fullfile(projRoot,'quality'));
addpath(fullfile(projRoot,'demo'));
addpath(fullfile(projRoot,'classification'));
grades = [0 1 2 3];   % 3 = Severe+Prolif merged truth
T = readtable(fullfile(projRoot,'dataset','APTOS','train.csv'));
S = load('C:/Users/SHIVANYA SALES/Desktop/DR tejas/HandheldDR/results/handheld_resnet50_aptosidrid4.mat','trainedNetHandheld');
trainedNet = S.trainedNetHandheld;
fprintf('Model: handheld R50 aptosidrid4 (live swap target).\n');
results = struct('file',{},'trueGrade',{},'qualityStatus',{}, ...
    'predictedClass',{},'predGrade',{},'confidence',{},'calConf',{}, ...
    'referable',{},'referCorrect',{},'timeSec',{},'passed',{},'error',{});
nPass = 0; nGrade = 0; nRef = 0; nRefOk = 0; tTotal = 0;
for g = grades
    if g == 3
        rows = T(T.diagnosis == 3 | T.diagnosis == 4, :);
    else
        rows = T(T.diagnosis == g, :);
    end
    idx = unique(round(linspace(1, height(rows), min(nPerGrade, height(rows)))));
    fprintf('\n[grade %d] testing %d raw images...\n', g, numel(idx));
    for j = 1:numel(idx)
        fn = char(string(rows.id_code(idx(j))) + ".png");
        R = struct('file',string(fn),'trueGrade',g,'qualityStatus',"", ...
            'predictedClass',"",'predGrade',NaN,'confidence',NaN,'calConf',NaN, ...
            'referable',false,'referCorrect',false,'timeSec',NaN,'passed',false,'error',"");
        try
            t0 = tic;
            I = imread(fullfile(projRoot,'dataset','APTOS','train_images',fn));
            res = screenFundusImage(I, trainedNet, 'SecondOpinion', true);
            R.timeSec = toc(t0);
            R.qualityStatus = string(res.qualityStatus);
            R.predictedClass = string(res.predictedClass);
            R.predGrade = res.ICDRGrade;
            R.confidence = res.modelScorePercent;
            R.calConf = res.calScorePercent;
            R.referable = logical(res.referableDR);
            R.passed = true; nPass = nPass + 1;
            if R.qualityStatus ~= "REJECT"
                if R.predGrade == g, nGrade = nGrade + 1; end
                refTrue = g >= 2;
                if R.referable == refTrue, nRefOk = nRefOk + 1; end
                nRef = nRef + 1;
            end
            tTotal = tTotal + R.timeSec;
            fprintf('  PASS %-22s true=%d pred=%s(%g) conf=%5.1f cal=%5.1f %s %.1fs\n', ...
                fn, g, char(R.predictedClass), R.predGrade, R.confidence, R.calConf, ...
                char(R.qualityStatus), R.timeSec);
        catch ME
            R.error = string(ME.message);
            fprintf('  FAIL %-22s %s\n', fn, char(R.error));
        end
        R.referCorrect = (R.referable == (g >= 2)) & R.passed;
        results(end+1) = R; %#ok<AGROW>
    end
end
n = numel(results);
fprintf('\n========================================\n');
fprintf('          RAW SMOKE SUMMARY (R50)\n');
fprintf('========================================\n');
fprintf('Images tested : %d\nPassed : %d\n', n, nPass);
if nRef > 0
    fprintf('Exact-grade (merged) acc : %.1f%%\n', 100*nGrade/max(nRef,1));
    fprintf('Referable acc            : %.1f%%\n', 100*nRefOk/max(nRef,1));
end
fprintf('Avg time/image: %.1fs\n', tTotal/max(nPass,1));
repDir = fullfile(projRoot,'reports');
if ~isfolder(repDir), mkdir(repDir); end
csvPath = fullfile(repDir, sprintf('raw_smoke_%d.csv', n));
fid = fopen(csvPath,'w');
fprintf(fid,'file,trueGrade,qualityStatus,predictedClass,predGrade,confidence,calConf,referable,timeSec,passed,error\n');
for k = 1:n
    fprintf(fid,'%s,%d,%s,%s,%g,%.2f,%.2f,%s,%.2f,%d,"%s"\n', ...
        char(results(k).file), results(k).trueGrade, char(results(k).qualityStatus), ...
        char(results(k).predictedClass), results(k).predGrade, results(k).confidence, ...
        results(k).calConf, char(string(results(k).referable)), ...
        results(k).timeSec, results(k).passed, char(results(k).error));
end
fclose(fid);
fprintf('CSV saved: %s\n', csvPath);
end
