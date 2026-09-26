function R = evalVesselEpoch(net, projRoot, ids, useGPU, tag)
% evalVesselEpoch — per-epoch, per-image scorer for VES P2B Arm A.
% Inference matches training: resize to 256, predict, argmax ch2, upsample
% to native 584x565 with nearest (same as demo/runVesselSegmentation.m:30-41).
% Reports per image: dFix (@0.5), dBest (sweep 0.05:0.05:0.9), vessel% pred
% vs GT, out-of-FOV FP fraction, thin-recall (bwmorph thin Inf, tol 2px).
% Thin-recall is reported JOINTLY with vessel%/FP, never alone.
% R: struct array with fields id, dFix, dBest, bestThr, predFrac, gtFrac, fovFpFrac, thinRecall.
ths = 0.05:0.05:0.9;
R = struct('id', {}, 'dFix', {}, 'dBest', {}, 'bestThr', {}, ...
    'predFrac', {}, 'gtFrac', {}, 'fovFpFrac', {}, 'thinRecall', {});
for k = 1:numel(ids)
    id = ids(k);
    ipath = fullfile(projRoot, 'dataset/DRIVE/training/images', sprintf('%d_training.tif', id));
    mpath = fullfile(projRoot, 'dataset/DRIVE/training/1st_manual', sprintf('%d_manual1.gif', id));
    fpath = fullfile(projRoot, 'dataset/DRIVE/training/mask', sprintf('%d_training_mask.gif', id));
    I = imread(ipath);
    [H, W, ~] = size(I);
    GT = imread(mpath) > 0;
    FOV = imread(fpath) > 0;
    I256 = imresize(I, [256 256]);
    if useGPU
        Xd = gpuArray(dlarray(single(I256), 'SSC'));
    else
        Xd = dlarray(single(I256), 'SSC');
    end
    Y = forward(net, Xd);
    P256 = single(extractdata(gather(Y(:, :, 2))));
    % score at native res: upsample probability then threshold sweep
    Pfull = imresize(P256, [H W], 'bilinear');
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
    % out-of-FOV FP: predicted vessel outside FOV circle / num predicted
    fovFpFrac = nnz(Qf & ~FOV) / max(nnz(Qf), 1);
    % thin recall: skeletonize GT vessels, hit if predicted within 2px
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
