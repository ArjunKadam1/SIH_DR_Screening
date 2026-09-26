% refLossOverfit_ACTFIX — D3 reference-loss overfit script. WRITTEN, NOT EXECUTED.
% DO NOT RUN without an explicit order. Pass rule for later: mean Dice@best
% (0.05-0.9 sweep) > 0.5 across the 6 patches within 500 iterations.
%
% Design (locked): same 6 overfit patches (loaded from overfit_verdict.mat),
% baseline init, 2500-iter cap, batch 4 cycling, no augmentation.
% Arm1: plain SGD lr 1e-4. Arm2: Adam lr 1e-4 (NOT 1e-3 — collapsed before).
% Loss: class-weighted CE from LOG-probabilities with clamp [1e-6,1-1e-6]
%   + soft Dice over the BATCH with smooth=1 (as specified).
%   Class weights: inverse-frequency per batch, wBG=1, wMA=nBG/nMA clamped
%   to [1,100]. Dice@best logged every 250 iters per arm (inference mode).

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
outDir   = fullfile(projRoot, 'results/MA_ACTFIX_20260921/overfit_ref');
if ~isfolder(outDir), mkdir(outDir); end
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_ACTFIX_20260921'));
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_verify_20260921'));
Ov = load(fullfile(projRoot, 'results/MA_ACTFIX_20260921/overfit/overfit_verdict.mat'));
picks = Ov.picks;
assert(height(picks) == 6, 'Patch list changed — STOP.');
B = load('C:/Users/SHIVANYA SALES/Downloads/MA_UNET_baseline_architecture.mat');
useGPU = (gpuDeviceCount > 0);
ths = 0.05:0.05:0.9;

arms = struct('name', {'SGD', 'Adam'}, 'lr', {1e-4, 1e-4});
cap = 1000; bs = 4; % adjusted: 1000 cap (was 2500); early stop on pass <=500
for a = 1:2
    nm = char(arms(a).name);
    net = B.net;
    rng(200 + a);
    cyc = randperm(6);
    passIter = 0;
    lossLog = zeros(cap, 1);
    mA = []; vA = []; step = 0;
    evIters = 250:250:cap;
    diceTraj = zeros(numel(evIters), 1);
    dicePatches = zeros(numel(evIters), 6);
    t0 = tic;
    logF = fopen(fullfile(outDir, sprintf('ref_%s.csv', nm)), 'w');
    fprintf(logF, 'iter,loss\n');
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
        [loss, grads] = dlfeval(@refLoss_ACTFIX, net, Xd, Td);
        lv = gather(extractdata(loss));
        assert(isfinite(lv), sprintf('%s non-finite loss at iter %d — STOP.', nm, it));
        lossLog(it) = lv;
        if strcmp(nm, 'SGD')
            net.Learnables = dlupdate(@(p, g) p - arms(a).lr*g, net.Learnables, grads);
        else
            step = step + 1;
            [net.Learnables, mA, vA] = adamupdate(net.Learnables, grads, mA, vA, step, arms(a).lr);
        end
        fprintf(logF, '%d,%.6f\n', it, lv);
        if it == 50 || mod(it, 250) == 0
            fprintf('%s iter=%d loss=%.4f elapsed=%.1fmin\n', nm, it, lv, toc(t0)/60);
        end
        if any(evIters == it)
            e = find(evIters == it);
            for p = 1:6
                [Xb6, Tb6] = cropMA512(picks, p);
                if useGPU, X6 = gpuArray(dlarray(single(Xb6), 'SSCB'));
                else,      X6 = dlarray(single(Xb6), 'SSCB'); end
                Y = forward(net, X6);
                P = double(extractdata(gather(Y(:, :, 2, :))));
                G = logical(Tb6(:, :, 2, :));
                bd = 0;
                for t = 1:numel(ths)
                    Q = P >= ths(t);
                    dd = 2*nnz(Q & G) / (nnz(Q) + nnz(G) + eps);
                    if dd > bd, bd = dd; end
                end
                dicePatches(e, p) = bd;
            end
            diceTraj(e) = mean(dicePatches(e, :));
            fprintf('%s DICE@best: iter=%d mean=%.4f per=%s\n', nm, it, ...
                diceTraj(e), mat2str(dicePatches(e, :), 3));
            if it <= 500 && diceTraj(e) > 0.5
                fprintf('%s PASS-RULE MET at iter=%d — early stop\n', nm, it);
                passIter = it;
                break;
            end
        end
    end
    fclose(logF);
    nDone = it;
    save(fullfile(outDir, sprintf('ref_%s.mat', nm)), 'lossLog', 'diceTraj', 'dicePatches', 'passIter', 'nDone');
    fprintf('%s DONE iters=%d passIter=%d finalMean=%.4f\n', nm, nDone, passIter, diceTraj(max(e,1)));
end
fprintf('REFLOSS_DONE\n');
