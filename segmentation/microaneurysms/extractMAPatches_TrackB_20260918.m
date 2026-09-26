% extractMAPatches_TrackB_20260918
% TRACK B ONLY — MA patch extraction (Phases 4). EXPERIMENTAL.
% This module is experimental. Never describe it as clinically validated.
% Generates patches separately per Phase-3 split (never from full set).
% Positive: ALL patches with >=1 MA pixel. Negatives capped at 3:1.
% Native resolution only (no downsampling) so lesion scale is preserved.
% Touch paths: segmentation/microaneurysms/patches/* and
%   results/MA/MA_patch_dataset_checkpoint.mat only.
% Never touches HandheldDR/, classification checkpoints, or Track A files.

projRoot  = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
splitFile = fullfile(projRoot,'results/MA/MA_split_checkpoint.mat');
outCkpt   = fullfile(projRoot,'results/MA/MA_patch_dataset_checkpoint.mat');

patchSize = 256;
stride    = 128;   % consistent stride, 50% overlap; recorded in checkpoint
negPosCap = 3;     % max negatives per positive
sampleSeed = 42;

fprintf('LOAD_SPLIT=%s EXISTS=%d\n', splitFile, isfile(splitFile));
assert(isfile(splitFile), 'Phase-3 split checkpoint missing — STOP.');
S = load(splitFile);
assert(isfield(S,'train_ids') && isfield(S,'val_ids') && isfield(S,'test_ids'), ...
    'Split checkpoint lacks ID lists — STOP.');

splitNames = {'train','val','test'};
splitIds   = {S.train_ids, S.val_ids, S.test_ids};

% Verify target dirs exist (or create images/masks subdirs) and are empty.
for s = 1:3
    for k = {'images','masks'}
        d = fullfile(projRoot,'segmentation/microaneurysms/patches',splitNames{s},k{1});
        if ~isfolder(d), mkdir(d); fprintf('MKDIR=%s\n', d); end
        f = dir(fullfile(d,'*.png'));
        fprintf('PRECHECK dir=%s npng=%d\n', d, numel(f));
        assert(isempty(f), sprintf('Target dir not empty — STOP (won''t overwrite): %s', d));
    end
end

% ---- PASS 1 (mask-only): count positives, collect negative coords ----
posCounts = cell(1,3); negCoords = cell(1,3); perImgPos = cell(1,3); perImgNegAvail = cell(1,3);
for s = 1:3
    ids = splitIds{s};
    nImg = numel(ids);
    pc = zeros(nImg,1); nc = zeros(nImg,1);
    ncoords = cell(nImg,1);
    for i = 1:nImg
        imgPath = char(ids(i));
        tok = regexp(imgPath,'IDRiD_\d+','match','once');
        assert(~isempty(tok), sprintf('Cannot parse image ID — STOP: %s', imgPath));
        if contains(imgPath,'a. Training Set')
            mskPath = fullfile(projRoot,'dataset/IDRiD/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/1. Microaneurysms',[tok '_MA.tif']);
        else
            mskPath = fullfile(projRoot,'dataset/IDRiD/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/1. Microaneurysms',[tok '_MA.tif']);
        end
        assert(isfile(imgPath), sprintf('Image missing — STOP: %s', imgPath));
        assert(isfile(mskPath), sprintf('Mask missing — STOP: %s', mskPath));
        info = imfinfo(imgPath);
        M = imread(mskPath);
        assert(isequal(size(M,1),info.Height) && isequal(size(M,2),info.Width), ...
            sprintf('Size mismatch — STOP: %s vs %s', imgPath, mskPath));
        u = unique(M(:)');
        assert(all(u==0|u==1), sprintf('Non-binary mask — STOP: %s values=%s', mskPath, mat2str(u)));
        Mb = M > 0;
        [H,W] = size(Mb);
        rs = unique([1:stride:(H-patchSize+1), H-patchSize+1]);
        cs = unique([1:stride:(W-patchSize+1), W-patchSize+1]);
        npos = 0; nlist = zeros(numel(rs)*numel(cs),2); nn = 0;
        for r = rs
            for c = cs
                if nnz(Mb(r:r+patchSize-1, c:c+patchSize-1)) >= 1
                    npos = npos + 1;
                else
                    nn = nn + 1; nlist(nn,:) = [r c];
                end
            end
        end
        nlist = nlist(1:nn,:);
        pc(i) = npos; nc(i) = nn; ncoords{i} = nlist;
    end
    posCounts{s} = pc; perImgPos{s} = pc; perImgNegAvail{s} = nc; negCoords{s} = ncoords;
    fprintf('PASS1 split=%s nImg=%d totalPosPatches=%d totalNegAvail=%d\n', splitNames{s}, nImg, sum(pc), sum(nc));
end

% ---- Sample negatives per split: quota = min(avail, 3 * totalPos) ----
rng(sampleSeed);
negKeep = cell(1,3); % per image kept negative coords
for s = 1:3
    totPos = sum(perImgPos{s});
    assert(totPos > 0, sprintf('Zero positives in %s — STOP.', splitNames{s}));
    % flatten negatives with image index
    allNeg = []; % [imgIdx r c]
    for i = 1:numel(negCoords{s})
        nl = negCoords{s}{i};
        if ~isempty(nl), allNeg = [allNeg; repmat(i,size(nl,1),1) nl]; end %#ok<AGROW>
    end
    quota = min(size(allNeg,1), negPosCap * totPos);
    sel = allNeg(randperm(size(allNeg,1), quota), :);
    nk = cell(numel(negCoords{s}),1);
    for i = 1:numel(nk), nk{i} = sel(sel(:,1)==i, 2:3); end
    negKeep{s} = nk;
    fprintf('SAMPLE split=%s totPos=%d negAvail=%d quota=%d ratio=%.4f\n', ...
        splitNames{s}, totPos, size(allNeg,1), quota, quota/totPos);
end

% ---- PASS 2: write PNGs ----
stats = struct();
for s = 1:3
    ids = splitIds{s};
    nImg = numel(ids);
    totPos = 0; totNeg = 0; mapix = uint64(0); totpix = uint64(0);
    perImgTot = zeros(nImg,1);
    for i = 1:nImg
        imgPath = char(ids(i));
        tok = regexp(imgPath,'IDRiD_\d+','match','once');
        if contains(imgPath,'a. Training Set')
            mskPath = fullfile(projRoot,'dataset/IDRiD/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/1. Microaneurysms',[tok '_MA.tif']);
        else
            mskPath = fullfile(projRoot,'dataset/IDRiD/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/1. Microaneurysms',[tok '_MA.tif']);
        end
        I = imread(imgPath); M = imread(mskPath) > 0;
        [H,W,~] = size(I);
        rs = unique([1:stride:(H-patchSize+1), H-patchSize+1]);
        cs = unique([1:stride:(W-patchSize+1), W-patchSize+1]);
        base = tok;
        imgOut = fullfile(projRoot,'segmentation/microaneurysms/patches',splitNames{s},'images');
        mskOut = fullfile(projRoot,'segmentation/microaneurysms/patches',splitNames{s},'masks');
        np = 0;
        for r = rs
            for c = cs
                Mp = M(r:r+patchSize-1, c:c+patchSize-1);
                if nnz(Mp) >= 1
                    Ip = I(r:r+patchSize-1, c:c+patchSize-1, :);
                    nm = sprintf('%s_r%04d_c%04d_pos.png', base, r, c);
                    imwrite(Ip, fullfile(imgOut, nm));
                    imwrite(uint8(Mp), fullfile(mskOut, nm));
                    np = np + 1; totPos = totPos + 1;
                    mapix = mapix + uint64(nnz(Mp)); totpix = totpix + uint64(patchSize*patchSize);
                end
            end
        end
        nk = negKeep{s}{i};
        nneg = 0;
        for k = 1:size(nk,1)
            r = nk(k,1); c = nk(k,2);
            Ip = I(r:r+patchSize-1, c:c+patchSize-1, :);
            Mp = M(r:r+patchSize-1, c:c+patchSize-1);
            nm = sprintf('%s_r%04d_c%04d_neg.png', base, r, c);
            imwrite(Ip, fullfile(imgOut, nm));
            imwrite(uint8(Mp), fullfile(mskOut, nm));
            nneg = nneg + 1; totNeg = totNeg + 1;
            mapix = mapix + uint64(nnz(Mp)); totpix = totpix + uint64(patchSize*patchSize);
        end
        perImgTot(i) = np + nneg;
        assert(np == perImgPos{s}(i), 'Positive recount mismatch — STOP.');
    end
    ratio = totNeg / totPos;
    posPixPct = 100 * double(mapix) / double(totpix);
    stats.(splitNames{s}) = struct('total', totPos+totNeg, 'pos', totPos, 'neg', totNeg, ...
        'negPosRatio', ratio, 'posPixelPct', posPixPct, ...
        'perImgMean', mean(perImgTot), 'perImgMin', min(perImgTot), 'perImgMax', max(perImgTot), ...
        'perImgTot', perImgTot);
    fprintf('SAVED split=%s total=%d pos=%d neg=%d ratio=%.4f posPixPct=%.6f perImgMean=%.2f range=[%d,%d]\n', ...
        splitNames{s}, totPos+totNeg, totPos, totNeg, ratio, posPixPct, mean(perImgTot), min(perImgTot), max(perImgTot));
end

patchSize_out = patchSize; stride_out = stride;
save(outCkpt, 'stats', 'patchSize_out', 'stride_out', 'negPosCap', 'sampleSeed', 'splitFile');
fprintf('SAVED_CHECKPOINT=%s\n', outCkpt);
