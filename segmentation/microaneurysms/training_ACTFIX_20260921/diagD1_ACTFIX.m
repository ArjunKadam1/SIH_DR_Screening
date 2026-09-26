% diagD1_ACTFIX — D1 evaluation-path check.
% Retrains the SGD overfit arm identically (seed rng(101), same 6 patches,
% batch 4 cycling, 2500 iters) and SAVES the final net (prior run did not).
% Then on the 6 patches: inference Dice@best (plain forward, per patch)
% vs training-mode Dice@best (forward under dlfeval = dropout active),
% fixed batches of 4 ([1:4],[5:6]), averaged over 5 passes, seeds logged.
% Rules: >=0.1 gap -> artifact suspected; within 0.05 -> path cleared.
% (BN=0 verified, so only the 2 dropout layers L21/L27 can differ.)

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
outDir   = fullfile(projRoot, 'results/MA_ACTFIX_20260921/diag');
if ~isfolder(outDir), mkdir(outDir); end
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_ACTFIX_20260921'));
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_verify_20260921'));
manDir = 'C:/Users/SHIVANYA SALES/Downloads/MA_Patch_Manifests';
T = load(fullfile(manDir, 'MA_train_manifest.mat')); m = T.manifest;
B = load('C:/Users/SHIVANYA SALES/Downloads/MA_UNET_baseline_architecture.mat');
Ov = load(fullfile(projRoot, 'results/MA_ACTFIX_20260921/overfit/overfit_verdict.mat'));
picks = Ov.picks;
useGPU = (gpuDeviceCount > 0);
ths = 0.05:0.05:0.9;

net = B.net;
rng(101); cyc = randperm(6);
lr = 1e-4; bs = 4; cap = 2500;
t0 = tic;
for it = 1:cap
    idx = cyc(mod((it-1)*bs, 6)+1:min(mod((it-1)*bs, 6)+bs, 6));
    if numel(idx) < bs, idx = [idx, cyc(1:bs-numel(idx))]; end
    [Xb, Tb] = cropMA512(picks, idx);
    if useGPU
        Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
        Td = gpuArray(dlarray(single(Tb), 'SSCB'));
    else
        Xd = dlarray(single(Xb), 'SSCB');
        Td = dlarray(single(Tb), 'SSCB');
    end
    [loss, grads] = dlfeval(@MA_modelGradients_ACTFIX_20260921, net, Xd, Td);
    net.Learnables = dlupdate(@(p, g) p - lr*g, net.Learnables, grads);
    if mod(it, 500) == 0
        fprintf('RETRAIN iter=%d loss=%.4f elapsed=%.1fmin\n', it, gather(extractdata(loss)), toc(t0)/60);
    end
end
save(fullfile(outDir, 'diagD1_finalSGDnet.mat'), 'net');
fprintf('RETRAIN_DONE\n');

% inference mode: plain forward per patch
dInf = zeros(6, 1);
for p = 1:6
    [Xb, Tb] = cropMA512(picks, p);
    if useGPU, Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
    else,      Xd = dlarray(single(Xb), 'SSCB'); end
    Y = forward(net, Xd);
    P = double(extractdata(gather(Y(:, :, 2, :))));
    G = logical(Tb(:, :, 2, :));
    bd = 0;
    for t = 1:numel(ths)
        Q = P >= ths(t);
        dd = 2*nnz(Q & G) / (nnz(Q) + nnz(G) + eps);
        if dd > bd, bd = dd; end
    end
    dInf(p) = bd;
end

% training mode: forward under dlfeval (dropout active), batches of 4, 5 passes
dTr = zeros(5, 6);
for pass = 1:5
    rng(7000 + pass);
    if useGPU, parallel.gpu.rng(7000 + pass); end
    for b = 1:2
        idx = (b-1)*4+1:min(b*4, 6);
        [Xb, Tb] = cropMA512(picks, idx);
        if useGPU
            Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
        else
            Xd = dlarray(single(Xb), 'SSCB');
        end
        Y = dlfeval(@fwdOnly, net, Xd);
        Pv = double(extractdata(gather(Y)));
        for j = 1:numel(idx)
            G = logical(Tb(:, :, 2, j));
            bd = 0;
            for t = 1:numel(ths)
                Q = Pv(:, :, 2, j) >= ths(t);
                dd = 2*nnz(Q & G) / (nnz(Q) + nnz(G) + eps);
                if dd > bd, bd = dd; end
            end
            dTr(pass, idx(j)) = bd;
        end
    end
    fprintf('TRAINMODE pass=%d mean=%.4f\n', pass, mean(dTr(pass, :)));
end
fprintf('INFER mean=%.4f per=%s\n', mean(dInf), mat2str(dInf', 4));
fprintf('TRAINMODE means=%s overall=%.4f\n', mat2str(mean(dTr, 2)', 4), mean(dTr(:)));
gap = abs(mean(dTr(:)) - mean(dInf));
fprintf('GAP=%.4f (rule: >=0.1 artifact; <=0.05 cleared)\n', gap);
save(fullfile(outDir, 'diagD1.mat'), 'dInf', 'dTr', 'gap');
fprintf('D1_DONE\n');

function Y = fwdOnly(net, X)
Y = forward(net, X);
end
