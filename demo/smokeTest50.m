function results = smokeTest50(nPerClass)
%% SMOKETEST50 - headless end-to-end smoke test over 50 fundus images.
% Stratified sample: 10 images x 5 DR classes (No_DR/Mild/Moderate/Severe/
% Proliferate_DR), evenly spaced through each sorted folder listing so the
% sample is deterministic and covers the folder.
%
% Runs the REAL demo pipeline per image (quality gate -> screenFundusImage
% with default second-opinion ON -> result struct validation). Headless:
% no figures, no PDFs - pure pipeline correctness + timing.
%
% Usage: results = smokeTest50()  (default 10/class = 50 images)
%        results = smokeTest50(4) (4/class = 20 images, quick check)
%
% Writes: reports/smoke_test_50.csv (per-image results)
% Returns: struct array with fields file,trueLabel,trueGrade,qualityStatus,
%   predictedClass,predGrade,confidence,referable,timeSec,passed,error.

if nargin < 1 || isempty(nPerClass)
    nPerClass = 10;
end

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot,'preprocessing'));
addpath(fullfile(projectRoot,'quality'));
addpath(fullfile(projectRoot,'demo'));
addpath(fullfile(projectRoot,'classification'));

% Phase-2 swap (2026-09-25): smoke exercises the live R50 4-class grader.
% Severe + Proliferate_DR folders both map to merged truth grade 3.
% ROLLBACK: restore 5-class list/grades + trained_resnet18.mat load.
classes = ["No_DR","Mild","Moderate","Severe","Proliferate_DR"];
trueGrades = [0 1 2 3 3];

fprintf('========================================\n');
fprintf('   DR Screening - Smoke Test (%d images)\n', nPerClass*numel(classes));
fprintf('========================================\n');

S = load('C:/Users/SHIVANYA SALES/Desktop/DR tejas/HandheldDR/results/handheld_resnet50_aptosidrid4.mat','trainedNetHandheld');
trainedNet = S.trainedNetHandheld;
fprintf('Model loaded successfully.\n');

results = struct('file',{},'trueLabel',{},'trueGrade',{},'qualityStatus',{}, ...
    'predictedClass',{},'predGrade',{},'confidence',{},'referable',{}, ...
    'timeSec',{},'passed',{},'error',{});

nPass = 0; nCorrect = 0; nGraded = 0; tTotal = 0;
for c = 1:numel(classes)
    d = dir(fullfile(projectRoot,'dataset',classes(c),'*.png'));
    if numel(d) < nPerClass
        error('smokeTest50:tooFewImages','%s has %d images, need %d', ...
            classes(c), numel(d), nPerClass);
    end
    idx = unique(round(linspace(1,numel(d),nPerClass))); % deterministic spread
    fprintf('\n[%s] testing %d images...\n', classes(c), numel(idx));
    for j = 1:numel(idx)
        f = d(idx(j)).name;
        R = struct('file',string(f),'trueLabel',classes(c), ...
            'trueGrade',trueGrades(c),'qualityStatus',"" , ...
            'predictedClass',"",'predGrade',NaN,'confidence',NaN, ...
            'referable',false,'timeSec',NaN,'passed',false,'error',"");
        try
            t0 = tic;
            I = imread(fullfile(d(idx(j)).folder,f));
            % Same pipeline as runDRDemoV2 (defaults: second-opinion ON)
            res = screenFundusImage(I, trainedNet, 'SecondOpinion', true);
            R.timeSec = toc(t0);
            % Validate result struct
            assert(isfield(res,'predictedClass') && isfield(res,'ICDRGrade'), ...
                'result struct missing classification fields');
            assert(res.ICDRGrade >= 0 && res.ICDRGrade <= 4, ...
                'ICDR grade out of range: %g', res.ICDRGrade);
            assert(res.modelScorePercent > 0 && res.modelScorePercent <= 100, ...
                'confidence out of range: %g', res.modelScorePercent);
            R.qualityStatus = string(res.qualityStatus);
            R.predictedClass = string(res.predictedClass);
            R.predGrade = res.ICDRGrade;
            R.confidence = res.modelScorePercent;
            R.referable = logical(res.referableDR);
            R.passed = true;
            nPass = nPass + 1;
            if ~strcmpi(R.qualityStatus,"REJECT")
                nGraded = nGraded + 1;
                if R.predGrade == R.trueGrade, nCorrect = nCorrect + 1; end
            end
            tTotal = tTotal + R.timeSec;
            fprintf('  PASS %-22s true=%d pred=%s(%d) conf=%6.2f%% %s %.1fs\n', ...
                f, R.trueGrade, R.predictedClass, R.predGrade, R.confidence, ...
                R.qualityStatus, R.timeSec);
        catch ME
            R.error = string(ME.message);
            fprintf('  FAIL %-22s %s\n', f, ME.message);
        end
        results(end+1) = R; %#ok<AGROW>
    end
end

% --- Summary ---
n = numel(results);
fprintf('\n========================================\n');
fprintf('          SMOKE TEST SUMMARY\n');
fprintf('========================================\n');
fprintf('Images tested : %d\n', n);
fprintf('Passed        : %d (%.1f%%)\n', nPass, 100*nPass/n);
fprintf('Failed        : %d\n', n-nPass);
fprintf('Graded        : %d (non-rejected)\n', nGraded);
if nGraded > 0
    fprintf('Exact-grade accuracy : %.1f%% (%d/%d)\n', 100*nCorrect/nGraded, nCorrect, nGraded);
end
fprintf('Avg time/image: %.1fs | Total: %.1fs\n', tTotal/max(nPass,1), tTotal);
fprintf('========================================\n');

% --- CSV ---
repDir = fullfile(projectRoot,'reports');
if ~isfolder(repDir), mkdir(repDir); end
csvPath = fullfile(repDir, sprintf('smoke_test_%d.csv', n));
fid = fopen(csvPath,'w');
fprintf(fid,'file,trueLabel,trueGrade,qualityStatus,predictedClass,predGrade,confidence,referable,timeSec,passed,error\n');
for k = 1:n
    fprintf(fid,'%s,%s,%d,%s,%s,%g,%.2f,%s,%.2f,%d,"%s"\n', ...
        results(k).file, results(k).trueLabel, results(k).trueGrade, ...
        results(k).qualityStatus, results(k).predictedClass, results(k).predGrade, ...
        results(k).confidence, string(results(k).referable), ...
        results(k).timeSec, results(k).passed, results(k).error);
end
fclose(fid);
fprintf('CSV saved: %s\n', csvPath);

if nPass < n
    warning('smokeTest50:failures','%d of %d images FAILED - see output above.', n-nPass, n);
else
    fprintf('SMOKE TEST: ALL %d IMAGES PASSED.\n', n);
end
end
