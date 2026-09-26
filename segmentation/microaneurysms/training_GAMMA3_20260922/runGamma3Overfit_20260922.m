% runGamma3Overfit_20260922 — NEW (branch MA_GAMMA3_20260922).
% 6-patch overfit (same picks via overfit_verdict.mat), baseline init,
% SGD lr 1e-4 + Adam lr 1e-4 arms (Adam 1e-3 already diverged on this loss
% family — documented, not repeated), batch 4 cycling rng(100+a) identical
% to D3 overfit (isolates the loss change), 1000-iter cap, Dice@best sweep
% every 250. Pass: mean Dice@best > 0.5. Divergence watch: loss>5 x50
% consecutive -> stop arm. No augmentation, no test set.
projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
outDir   = fullfile(projRoot, 'results/MA_GAMMA3_20260922');
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_GAMMA3_20260922'));
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_verify_20260921'));
Ov = load(fullfile(projRoot, 'results/MA_ACTFIX_20260921/overfit/overfit_verdict.mat'));
picks = Ov.picks;
assert(height(picks) == 6, 'Patch list changed — STOP.');
B = load('C:/Users/SHIVANYA SALES/Downloads/MA_UNET_baseline_architecture.mat');
useGPU = (gpuDeviceCount > 0);
ths = 0.05:0.05:0.9;

arms = struct('name', {'SGD', 'Adam'}, 'lr', {1e-4, 1e-4});
cap = 1000; bs = 4;
V = struct();
for a = 1:2
    nm = char(arms(a).name);
    net = B.net;
    rng(100 + a);
    cyc = randperm(6);
    lossLog = zeros(cap, 1);
    mA = []; vA = []; step = 0; over5 = 0; stopped = 0;
    evIters = 250:250:cap;
    diceTraj = zeros(numel(evIters), 1);
    dicePatches = zeros(numel(evIters), 6);
    t0 = tic;
    logF = fopen(fullfile(outDir, sprintf('g3_%s.csv', nm)), 'w');
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
        [loss, grads] = dlfeval(@refLoss_G3_20260922, net, Xd, Td);
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
        if lv > 5, over5 = over5 + 1; else, over5 = 0; end
        if over5 >= 50
            stopped = it;
            fprintf('%s DIVERGENCE WATCH: loss>5 x50 at iter %d — stopping arm.\n', nm, it);
            break;
        end
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
        end
    end
    fclose(logF);
    nDone = it;
    eDone = find(diceTraj > 0, 1, 'last');
    if isempty(eDone)
        fprintf('%s DONE iters=%d stopped=%d (no eval reached)\n', nm, nDone, stopped);
    else
        fprintf('%s DONE iters=%d stopped=%d finalMean=%.4f finalMin=%.4f\n', ...
            nm, nDone, stopped, diceTraj(eDone), min(dicePatches(eDone, :)));
    end
    save(fullfile(outDir, sprintf('g3_%s.mat', nm)), 'lossLog', 'diceTraj', 'dicePatches', 'nDone', 'stopped');
end
fprintf('G3_DONE\n');
