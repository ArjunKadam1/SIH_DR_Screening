% diagD2_dice5 — P4 D2-style ordering gate (NO training, CPU only).
% Synthetic P cases on real DRIVE GT (id 21 1st_manual), exact dice5 math:
% perfect / nothing / half / inverted / sat-wrong.
% GATE (enforced): perfect < half < nothing < inverted AND sat-wrong finite.
% Prints PASS/FAIL per rule. On FAIL the build must STOP with no training.
projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
addpath(fullfile(projRoot, 'segmentation/vessels/training_VES_P2B'));
W0 = 10.4785; % fixed normalizer from P3 audit (16-train mean frac 0.08712)
G = single(imread(fullfile(projRoot, 'dataset/DRIVE/training/1st_manual/21_manual1.gif'))) > 0;
G = single(G);
fprintf('D2-dice5 GT vessel-frac=%.5f W0=%.4f\n', mean(G(:)), W0);
Gb = logical(G);
cases = struct('name', {'perfect','nothing','half','inverted','satwrong'}, ...
    'P', {G, zeros(size(G),'single'), 0.5*ones(size(G),'single'), ...
           single(~Gb), single(~Gb)});
% satwrong: exactly 0 at vessels, exactly 1 elsewhere (= inverted pattern,
% kept as separate case for the finite/clamp check)
names = {cases.name};
res = zeros(1, 5);
ok = true(1, 5);
for k = 1:5
    P = cases(k).P;
    e = 1e-6;
    Pc = min(max(P, e), 1 - e);
    logP = log(Pc); logN = log(1 - Pc);
    nV = sum(G(:)); nB = sum(1 - G(:));
    wV = min(max(nB / max(nV, 1), 1), 100);
    ce = -mean(G(:) .* (wV/W0) .* logP(:) + (1 - G(:)) .* logN(:));
    inter = sum(P(:) .* G(:));
    dice = (2*inter + 1) / (sum(P(:)) + sum(G(:)) + 1);
    di = 1 - dice;
    res(k) = ce + 5*di;
    ok(k) = isfinite(res(k));
    fprintf('D2-dice5 %-8s loss=%.4f (ce=%.4f di5=%.4f) finite=%d\n', ...
        names{k}, res(k), ce, 5*di, ok(k));
end
pass = ok(1)&ok(2)&ok(3)&ok(4)&ok(5) & ...
    (res(1) < res(3)) & (res(3) < res(2)) & (res(2) < res(4));
fprintf('D2-dice5 GATE perfect<half<nothing<inverted: %.4f<%.4f<%.4f<%.4f -> %s\n', ...
    res(1), res(3), res(2), res(4), string(pass));
fprintf('D2-dice5 half-vs-nothing margin=%.4f\n', res(2) - res(3));
if pass
    fprintf('D2-dice5 GATE PASS — training authorized.\n');
else
    fprintf('D2-dice5 GATE FAIL — STOP, no training.\n');
end
