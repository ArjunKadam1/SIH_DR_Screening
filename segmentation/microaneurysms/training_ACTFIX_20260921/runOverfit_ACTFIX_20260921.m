% runOverfit_ACTFIX_20260921 — NEW (branch MA_ACTFIX_20260921, Step B).
% Overfit 6 ordinary-size-lesion TRAIN patches (exact eval crops, no aug).
% Arm1: plain SGD lr 1e-4 (same settings as epoch run). Arm2: Adam lr 1e-3
% (deliberate deviation, separate arm). Cap 2500 iters/arm.
% Log: loss/iter (CSV) + Dice@best-threshold (0.05-0.9 sweep) every 250.
% Plateau (locked): <1% change in 100-iter MA over last 300 iters.
% Pass (locked): mean Dice@best > 0.5 across the 6 (all six published;
% mean-pass with min<0.3 = weak pass with carry caveat).
% BN note: dlnetwork forward() in custom loops has no training-mode flag
% (trainnet manages BN internally), so the training-mode-BN secondary
% column is omitted as API-unsupported; verdicts rest on inference-mode
% forwards, identical to all prior analyses.

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
outDir   = fullfile(projRoot, 'results/MA_ACTFIX_20260921/overfit');
if ~isfolder(outDir), mkdir(outDir); end
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_ACTFIX_20260921'));
addpath(fullfile(projRoot, 'segmentation/microaneurysms/training_verify_20260921'));
manDir = 'C:/Users/SHIVANYA SALES/Downloads/MA_Patch_Manifests';
T = load(fullfile(manDir, 'MA_train_manifest.mat')); m = T.manifest;
B = load('C:/Users/SHIVANYA SALES/Downloads/MA_UNET_baseline_architecture.mat');
useGPU = (gpuDeviceCount > 0);

% ---- pick 6 ordinary-size lesions: near-median LesionPixels, 3+3 datasets ----
pos = m(m.IsPositive, :);
medLP = median(pos.LesionPixels);
pos.d = abs(double(pos.LesionPixels) - medLP);
picks = [];
for ds = ["IDRiD", "eOphtha"]
    sub = pos(pos.Dataset == ds, :);
    sub = sortrows(sub, 'd');
    picks = [picks; sub(1:3, :)]; %#ok<AGROW>
end
fprintf('OVERFIT PATCHES (medianLP=%.0f):\n', medLP);
for i = 1:6
    fprintf(' %d: ds=%s lesion=%d X=%d Y=%d %s\n', i, picks.Dataset(i), ...
        picks.LesionPixels(i), picks.X(i), picks.Y(i), picks.ImagePath(i));
end
assert(height(picks) == 6, 'Patch pick failed — STOP.');

ths = 0.05:0.05:0.9;
arms = struct('name', {'SGD', 'Adam'}, 'lr', {1e-4, 1e-3});
cap = 2500; bs = 4;
V = struct();
for a = 1:2
    nm = char(arms(a).name);
    net = B.net; % fresh init per arm
    rng(100 + a); % logged cycling order per arm
    cyc = randperm(6);
    lossLog = zeros(cap, 1);
    mA = []; vA = []; step = 0;
    evIters = 250:250:cap;
    diceTraj = zeros(numel(evIters), 1);
    dicePatches = zeros(numel(evIters), 6);
    t0 = tic;
    logF = fopen(fullfile(outDir, sprintf('overfit_%s.csv', nm)), 'w');
    fprintf(logF, 'iter,loss\n');
    for it = 1:cap
        idx = cyc(mod((it-1)*bs, 6)+1:min(mod((it-1)*bs, 6)+bs, 6));
        if numel(idx) < bs
            idx = [idx, cyc(1:bs-numel(idx))];
        end
        [Xb, Tb] = cropMA512(picks, idx);
        if useGPU
            Xd = gpuArray(dlarray(single(Xb), 'SSCB'));
            Td = gpuArray(dlarray(single(Tb), 'SSCB'));
        else
            Xd = dlarray(single(Xb), 'SSCB');
            Td = dlarray(single(Tb), 'SSCB');
        end
        [loss, grads] = dlfeval(@MA_modelGradients_ACTFIX_20260921, net, Xd, Td);
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
            fprintf('%s DICE@best250: iter=%d mean=%.4f per=%s\n', nm, it, ...
                diceTraj(e), mat2str(dicePatches(e, :), 3));
        end
    end
    fclose(logF);
    ma100 = movmean(lossLog, 100);
    plat = abs(ma100(end) - ma100(end-300)) / max(ma100(end-300), eps) < 0.01;
    V.(char(nm)) = struct('lossLog', lossLog, 'diceTraj', diceTraj, ...
        'dicePatches', dicePatches, 'plateaued', plat, ...
        'finalMean', diceTraj(end), 'finalMin', min(dicePatches(end, :)));
    fprintf('%s DONE plateaued=%d finalMean=%.4f finalMin=%.4f\n', ...
        nm, plat, diceTraj(end), min(dicePatches(end, :)));
    save(fullfile(outDir, sprintf('overfit_%s.mat', nm)), 'lossLog', 'diceTraj', 'dicePatches');
end

% ---- verdict table (locked) ----
gS = V.SGD.finalMean > 0.5; gA = V.Adam.finalMean > 0.5;
pS = V.SGD.plateaued; pA = V.Adam.plateaued;
fprintf('VERDICT SGD pass=%d plateaued=%d | Adam pass=%d plateaued=%d\n', gS, pS, gA, pA);
save(fullfile(outDir, 'overfit_verdict.mat'), 'V', 'picks');
fprintf('OVERFIT_DONE\n');
