function [dF, iF, dB, mMA, mBG] = eval50_STAB3(net, v, esel, useGPU)
% eval50_STAB3 — NEW (branch MA_STAB3_20260922).
% Fixed-threshold (@0.5) means + best-threshold-sweep means on 50 patches.
% Inference mode (plain forward), raw softmax ch2, no sigmoid.
ths = 0.05:0.05:0.9;
df = zeros(50, 1); iff = zeros(50, 1); db = zeros(50, 1);
sMA = 0; nMA = 0; sBG = 0; nBG = 0;
addpath('C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening/segmentation/microaneurysms/training_verify_20260921');
for k = 1:50
    [Xb, Tb] = cropMA512(v, esel(k));
    if useGPU, Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
    else,      Xd = dlarray(single(Xb), 'SSCB'); end
    Y = forward(net, Xd);
    P = double(extractdata(gather(Y(:, :, 2, :))));
    G = logical(Tb(:, :, 2, :));
    sMA = sMA + sum(P(G)); nMA = nMA + nnz(G);
    sBG = sBG + sum(P(~G)); nBG = nBG + nnz(~G);
    Qf = P >= 0.5;
    inter = nnz(Qf & G);
    df(k) = 2*inter / (nnz(Qf) + nnz(G) + eps);
    iff(k) = inter / (nnz(Qf | G) + eps);
    bd = 0;
    for t = 1:numel(ths)
        Q = P >= ths(t);
        dd0 = 2*nnz(Q & G) / (nnz(Q) + nnz(G) + eps);
        if dd0 > bd, bd = dd0; end
    end
    db(k) = bd;
end
dF = mean(df); iF = mean(iff); dB = mean(db);
mMA = sMA / max(nMA, 1); mBG = sBG / max(nBG, 1);
end
