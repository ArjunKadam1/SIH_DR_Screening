function L = runLesionEvidence(I, maNet, vesNet, fovMask, opts)
% RUNLESIONEVEDENCE - shared MA + P3-A vessel inference for the new app.
%
% Usage:
%   L = runLesionEvidence(I, maNet, vesNet, fovMask)
%   L = runLesionEvidence(I, maNet, vesNet, fovMask, opts)
%
% Inputs:
%   I       - HxWx3 fundus image (uint8 or single). Full resolution, NOT 224.
%   maNet   - dlnetwork MA U-Net (MA_LR3E5 bestNet, 512x512), or [] to skip.
%   vesNet  - dlnetwork P3-A vessel net (armA_block2_iter80 net, 256x256),
%             or [] to skip.
%   fovMask - HxW logical FOV mask (uploaded by user), or [] if none.
%             Vessel output is masked by it when present; when absent the
%             branch still runs but reports full-frame density with a warning
%             note (upload a mask for honest FOV density).
%   opts    - struct, all optional:
%             .runMA (true), .runVessel (true),
%             .maThr (0.5, matches manualTestRun_IDRiD54 + train eval),
%             .vesMode ('percentile' default | 'fixed'),
%             .vesThr (0.5, fixed-mode threshold = train/eval scoring),
%             .vesTargetFrac (0.09, percentile-mode vessel density, anchored
%               to train-GT mean ~8.7%% like runAttentionVesselDemo),
%             .vesMinArea (20, speck cleanup like runAttentionVesselDemo)
%
% Output L struct (always returned, never throws):
%   .maAvailable, .maMask (HxW logical), .maProbMap (HxW single),
%   .maCount (connected components), .maThr, .maTimeSec,
%   .vesselAvailable, .vesselMask, .vesselProbMap, .vesselFracFOV,
%   .vesselThr, .vesselTimeSec, .lesionNote
%
% Notes:
%   MA: 512/256 tiled forward->Y(:,:,2), averaged, >=maThr.
%       Pattern from training_LR3E5_20260922/manualTestRun_IDRiD54.m.
%   Vessel P3-A: resize-256 forward->ch2, thresholded, & FOV, bwareaopen.
%       Pattern from trainVesselP3_ArmA.m:128-133 and evalVesselEpoch.m.
%       Pooled val: dFix 0.3735 block2, thinRecall ~0.73.
%       Threshold: 'percentile' pins output to the densest vesTargetFrac
%       of the FOV domain (robust on APTOS, where fixed 0.5 washes out —
%       e.g. 50.99%% full-frame on 12e6e66c80a7.png); 'fixed' uses vesThr
%       (matches DRIVE train/eval scoring). Sort-based percentile uses
%       base MATLAB only (no Statistics Toolbox).
%   GPU used when available, CPU fallback otherwise. No figures here —
%   the app displays. Prototype labels: MA 0.42 val / vessel ~0.37 val
%   Dice on DRIVE 37-40, supporting evidence only, NOT clinical.

L = struct( ...
    'maAvailable', false, 'maMask', [], 'maProbMap', [], ...
    'maCount', 0, 'maThr', 0.5, 'maTimeSec', NaN, ...
    'vesselAvailable', false, 'vesselMask', [], 'vesselProbMap', [], ...
    'vesselFracFOV', NaN, 'vesselThr', 0.5, 'vesselTimeSec', NaN, ...
    'lesionNote', '');

if nargin < 5 || isempty(opts), opts = struct(); end
if ~isfield(opts, 'runMA'), opts.runMA = true; end
if ~isfield(opts, 'runVessel'), opts.runVessel = true; end
if ~isfield(opts, 'maThr'), opts.maThr = 0.5; end
if ~isfield(opts, 'vesMode'), opts.vesMode = 'percentile'; end
if ~isfield(opts, 'vesThr'), opts.vesThr = 0.5; end
if ~isfield(opts, 'vesTargetFrac'), opts.vesTargetFrac = 0.09; end
if ~isfield(opts, 'vesMinArea'), opts.vesMinArea = 20; end
L.maThr = double(opts.maThr);
L.vesselThr = double(opts.vesThr);  % overwritten below when percentile runs

notes = {};
useGPU = false;
try
    useGPU = (gpuDeviceCount > 0);
catch
    useGPU = false;
end

if isempty(I)
    L.lesionNote = 'Empty image.';
    return;
end
if size(I, 3) == 1
    I = repmat(I, 1, 1, 3);
end
[H, W, ~] = size(I);

% ---------------- MA branch ----------------
if opts.runMA && ~isempty(maNet)
    try
        t0 = tic;
        ps = 512; st = 256;
        % Pad small images symmetrically so tiling always works.
        padH = max(0, ps - H); padW = max(0, ps - W);
        if padH > 0 || padW > 0
            Ipad = padarray(I, [ceil(padH/2) ceil(padW/2)], 'symmetric', 'pre');
            Ipad = padarray(Ipad, [floor(padH/2) floor(padW/2)], 'symmetric', 'post');
        else
            Ipad = I;
        end
        [Hp, Wp, ~] = size(Ipad);
        rs = unique([1:st:(Hp-ps+1), Hp-ps+1]);
        cs = unique([1:st:(Wp-ps+1), Wp-ps+1]);
        rs(rs < 1) = []; cs(cs < 1) = [];
        acc = zeros(Hp, Wp, 'single');
        cnt = zeros(Hp, Wp, 'single');
        for r = rs
            for c = cs
                tile = im2single(Ipad(r:r+ps-1, c:c+ps-1, :));
                if useGPU
                    Xd = gpuArray(dlarray(tile, 'SSC'));
                else
                    Xd = dlarray(tile, 'SSC');
                end
                Y = forward(maNet, Xd);
                P = single(extractdata(gather(Y(:, :, 2))));
                acc(r:r+ps-1, c:c+ps-1) = acc(r:r+ps-1, c:c+ps-1) + P;
                cnt(r:r+ps-1, c:c+ps-1) = cnt(r:r+ps-1, c:c+ps-1) + 1;
            end
        end
        Ppad = acc ./ max(cnt, 1);
        % Crop padding back to original size.
        r0 = ceil(padH/2) + 1; c0 = ceil(padW/2) + 1;
        Pfull = Ppad(r0:r0+H-1, c0:c0+W-1);
        Q = Pfull >= opts.maThr;
        CC = bwconncomp(Q, 8);
        L.maAvailable = true;
        L.maMask = logical(Q);
        L.maProbMap = single(Pfull);
        L.maCount = CC.NumObjects;
        L.maTimeSec = toc(t0);
    catch ME
        notes{end+1} = sprintf('MA failed: %s', ME.message); %#ok<AGROW>
        L.maAvailable = false;
    end
elseif opts.runMA
    notes{end+1} = 'MA net not loaded.'; %#ok<AGROW>
end

% ---------------- Vessel P3-A branch ----------------
if opts.runVessel && ~isempty(vesNet)
    try
        t0 = tic;
        I256 = imresize(I, [256 256]);
        if useGPU
            Xd = gpuArray(dlarray(single(I256), 'SSC'));
        else
            Xd = dlarray(single(I256), 'SSC');
        end
        Y = forward(vesNet, Xd);
        P256 = single(extractdata(gather(Y(:, :, 2))));
        % Probability map at native res (bilinear) for heat display.
        Pfull = imresize(P256, [H W], 'bilinear');
        % FOV domain first: percentile pins density inside it.
        if ~isempty(fovMask)
            FOV = logical(fovMask);
            if ~isequal(size(FOV), [H W])
                FOV = imresize(FOV, [H W], 'nearest');
            end
            domMask = FOV;
            fovN = max(nnz(FOV), 1);
            hasFOV = true;
        else
            % Honest gate (upload-mask decision): report full-frame density
            % but flag it so the UI can show the banner.
            notes{end+1} = 'No FOV mask uploaded — vessel % is full-frame. Upload a mask for honest FOV density.'; %#ok<AGROW>
            domMask = true(H, W);
            fovN = H * W;
            hasFOV = false;
        end
        if strcmpi(opts.vesMode, 'percentile')
            Pv = sort(Pfull(domMask));
            if isempty(Pv)
                thrUsed = double(opts.vesThr);
            else
                ki = max(1, round((1 - opts.vesTargetFrac) * numel(Pv)));
                thrUsed = double(Pv(ki));
            end
        else
            thrUsed = double(opts.vesThr);
        end
        Q = Pfull >= thrUsed;
        if hasFOV
            Q = Q & FOV;
        end
        Q = bwareaopen(Q, opts.vesMinArea);
        L.vesselAvailable = true;
        L.vesselMask = logical(Q);
        L.vesselProbMap = single(Pfull);
        L.vesselThr = thrUsed;
        L.vesselFracFOV = 100 * nnz(Q) / max(fovN, 1);
        L.vesselTimeSec = toc(t0);
    catch ME
        notes{end+1} = sprintf('Vessel failed: %s', ME.message); %#ok<AGROW>
        L.vesselAvailable = false;
    end
elseif opts.runVessel
    notes{end+1} = 'Vessel net not loaded.'; %#ok<AGROW>
end

if isempty(notes)
    L.lesionNote = 'Prototype supporting evidence only — MA 0.42 val / P3-A vessel dFix 0.37 val. NOT clinical.';
else
    L.lesionNote = strjoin([notes, ...
        {'MA 0.42 val / P3-A vessel dFix 0.37 val. Prototype, NOT clinical.'}], ' ');
end
end
