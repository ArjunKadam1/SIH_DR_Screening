% buildMADatastore_TrackB_20260918
% TRACK B ONLY — Phase 5 datastore creation + verification. EXPERIMENTAL.
% This module is experimental. Never describe it as clinically validated.
% Creates MATLAB imageDatastore + pixelLabelDatastore pairs for
% train/val/test patches. Class definition: background=0, MA=1 (exact).
% Runs ALL verification checks; on ANY failure reports files and STOPS.
% Touch paths: results/MA/MA_datastore_checkpoint.mat only (read patches).
% Never touches HandheldDR/, classification checkpoints, or Track A files.

projRoot  = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
patchRoot = fullfile(projRoot,'segmentation/microaneurysms/patches');
outCkpt   = fullfile(projRoot,'results/MA/MA_datastore_checkpoint.mat');

assert(~isempty(which('pixelLabelDatastore')), ...
    'pixelLabelDatastore not found — STOP (Computer Vision Toolbox missing).');

classNames = ["background","MA"];
labelIDs   = [0 1];
fprintf('CLASSDEF names=%s ids=%s\n', strjoin(cellstr(classNames),','), mat2str(labelIDs));

splitNames = {'train','val','test'};
verifySeed = 7;
results = struct();

for s = 1:3
    sp = splitNames{s};
    imgDir = fullfile(patchRoot, sp, 'images');
    mskDir = fullfile(patchRoot, sp, 'masks');
    assert(isfolder(imgDir) && isfolder(mskDir), sprintf('Patch dir missing — STOP: %s', sp));

    imds = imageDatastore(imgDir);
    pxds = pixelLabelDatastore(mskDir, classNames, labelIDs);
    nImg = numel(imds.Files); nMsk = numel(pxds.Files);

    % CHECK 1: counts match
    check1 = (nImg == nMsk);
    fprintf('[%s] CHECK1 counts match: images=%d masks=%d -> %s\n', sp, nImg, nMsk, string(check1));

    % CHECKS 2-4 + pixel distribution: single full pass over RAW pngs
    % (numeric check; datastore readimage returns categorical, verified in CHECK6)
    expImgSize = [256 256 3];
    allImgOk = true; allMskMatch = true;
    badBinaryFiles = {}; badBinaryVals = [];
    bgPx = uint64(0); maPx = uint64(0);
    firstBadDim = '';
    for i = 1:nImg
        I = imread(imds.Files{i});
        Mraw = imread(pxds.Files{i});
        if ~isequal(size(I), expImgSize)
            allImgOk = false;
            if isempty(firstBadDim), firstBadDim = imds.Files{i}; end
        end
        if ~isequal(size(Mraw,1), size(I,1)) || ~isequal(size(Mraw,2), size(I,2))
            allMskMatch = false;
            if isempty(firstBadDim), firstBadDim = pxds.Files{i}; end
        end
        u = unique(double(Mraw(:))');
        if ~all(u==0 | u==1)
            badBinaryFiles{end+1} = pxds.Files{i}; %#ok<AGROW>
            badBinaryVals = union(badBinaryVals, u);
        end
        bgPx = bgPx + uint64(nnz(Mraw==0));
        maPx = maPx + uint64(nnz(Mraw==1));
    end
    check2 = allImgOk;
    check3 = allMskMatch;
    check4 = isempty(badBinaryFiles);
    fprintf('[%s] CHECK2 image dims consistent 256x256x3 -> %s\n', sp, string(check2));
    fprintf('[%s] CHECK3 mask dims match image dims -> %s\n', sp, string(check3));
    if check4
        fprintf('[%s] CHECK4 masks strictly binary (0/1 only) -> true\n', sp);
    else
        fprintf('[%s] CHECK4 masks strictly binary -> FALSE values=%s nBadFiles=%d e.g. %s\n', ...
            sp, mat2str(badBinaryVals), numel(badBinaryFiles), badBinaryFiles{1});
    end
    fprintf('[%s] CHECK5 pixel distribution: background=%d MA=%d total=%d maPct=%.6f\n', ...
        sp, bgPx, maPx, bgPx+maPx, 100*double(maPx)/double(bgPx+maPx));

    % CHECK 6: read 3 random samples via datastore readimage + confirm classes
    rng(verifySeed + s);
    sampIdx = randperm(nImg, min(3, nImg));
    check6 = true;
    for k = 1:numel(sampIdx)
        try
            Ii = readimage(imds, sampIdx(k));
            Mi = readimage(pxds, sampIdx(k));
            assert(iscategorical(Mi) && isequal(categories(Mi), {'background';'MA'}), 'unexpected mask classes');
            fprintf('[%s] CHECK6 sample %d/%d OK: %s size=%s maskSize=%s classes=%s\n', ...
                sp, k, numel(sampIdx), imds.Files{sampIdx(k)}, mat2str(size(Ii)), mat2str(size(Mi)), strjoin(categories(Mi)',','));
        catch ME
            check6 = false;
            fprintf('[%s] CHECK6 sample %d FAILED: %s err=%s\n', sp, k, imds.Files{sampIdx(k)}, ME.message);
        end
    end

    allPass = check1 && check2 && check3 && check4 && check6;
    fprintf('[%s] VERDICT allChecks=%s\n', sp, string(allPass));

    results.(sp) = struct('nImages', nImg, 'nMasks', nMsk, ...
        'checkCountsMatch', check1, 'checkImgDims', check2, ...
        'checkMaskDimsMatch', check3, 'checkBinary', check4, ...
        'checkReadSamples', check6, 'allPass', allPass, ...
        'bgPx', bgPx, 'maPx', maPx, ...
        'sampleIdx', sampIdx, 'firstBadDim', firstBadDim, ...
        'nBadBinary', numel(badBinaryFiles));

    assert(allPass, sprintf('Verification FAILED for %s — STOP. See log above.', sp));
end

cfg = struct('patchRoot', patchRoot, 'classNames', classNames, ...
    'labelIDs', labelIDs, 'verifySeed', verifySeed, ...
    'imageSubdir', 'images', 'maskSubdir', 'masks');
save(outCkpt, 'cfg', 'results');
fprintf('SAVED_CHECKPOINT=%s\n', outCkpt);
