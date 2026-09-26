function R = evalVesselEpochNative(net, projRoot, ids, useGPU, tag)
% evalVesselEpochNative — Arm B scorer: native 256 tiles stride 128,
% averaged at reconstruction (predictMAImageTiled pattern), no resize.
% Reports per image: dFix/dBest/bestThr/predFrac/gtFrac/fovFpFrac/thinRecall
% (bwmorph thin Inf, tol 2px; recall joint with vessel%/FP, never alone).
ps = 256; st = 128; ths = 0.05:0.05:0.9;
R = struct('id', {}, 'dFix', {}, 'dBest', {}, 'bestThr', {}, ...
    'predFrac', {}, 'gtFrac', {}, 'fovFpFrac', {}, 'thinRecall', {});
for k = 1:numel(ids)
    id = ids(k);
    I = imread(fullfile(projRoot, 'dataset/DRIVE/training/images', sprintf('%d_training.tif', id)));
    GT = imread(fullfile(projRoot, 'dataset/DRIVE/training/1st_manual', sprintf('%d_manual1.gif', id))) > 0;
    FOV = imread(fullfile(projRoot, 'dataset/DRIVE/training/mask', sprintf('%d_training_mask.gif', id))) > 0;
    [H, W, ~] = size(I);
    rs = unique([1:st:(H-ps), H-ps+1]); cs = unique([1:st:(W-ps), W-ps+1]);
    acc = zeros(H, W, 'single'); cnt = zeros(H, W, 'single');
    for r = rs
        for c = cs
            tile = im2single(I(r:r+ps-1, c:c+ps-1, :));
            if useGPU, Xd = gpuArray(dlarray(tile, 'SSC')); else, Xd = dlarray(tile, 'SSC'); end
            Y = forward(net, Xd);
            P = single(extractdata(gather(Y(:, :, 2))));
            acc(r:r+ps-1, c:c+ps-1) = acc(r:r+ps-1, c:c+ps-1) + P;
            cnt(r:r+ps-1, c:c+ps-1) = cnt(r:r+ps-1, c:c+ps-1) + 1;
        end
    end
    Pfull = acc ./ max(cnt, 1);
    Qf = Pfull >= 0.5;
    inter = nnz(Qf & GT);
    dFix = 2*inter / (nnz(Qf) + nnz(GT) + eps);
    bd = 0; bt = 0.5;
    for t = 1:numel(ths)
        Q = Pfull >= ths(t);
        dd0 = 2*nnz(Q & GT) / (nnz(Q) + nnz(GT) + eps);
        if dd0 > bd, bd = dd0; bt = ths(t); end
    end
    predFrac = nnz(Qf) / numel(Qf);
    gtFrac = nnz(GT) / numel(GT);
    fovFpFrac = nnz(Qf & ~FOV) / max(nnz(Qf), 1);
    skel = bwmorph(GT, 'thin', Inf);
    D = bwdist(Qf);
    skIdx = find(skel);
    thinRecall = nnz(D(skIdx) <= 2) / max(nnz(skel), 1);
    R(k).id = id; R(k).dFix = dFix; R(k).dBest = bd; R(k).bestThr = bt;
    R(k).predFrac = predFrac; R(k).gtFrac = gtFrac;
    R(k).fovFpFrac = fovFpFrac; R(k).thinRecall = thinRecall;
end
fprintf('[%s] eval %d imgs: mean dFix=%.4f dBest=%.4f predFrac=%.4f\n', ...
    tag, numel(ids), mean([R.dFix]), mean([R.dBest]), mean([R.predFrac]));
end
