% diagD2_focal — P5 D2-style ordering gate (NO training, CPU only).
% Synthetic P cases on real DRIVE GT (id 21 1st_manual), exact focal+dice5 math:
% perfect / nothing / half / inverted / satwrong (= inverted pattern, clamp check).
% GATE (enforced): perfect < half < nothing < inverted AND satwrong finite.
projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
addpath(fullfile(projRoot, 'segmentation/vessels/training_VES_P2B'));
G = single(imread(fullfile(projRoot, 'dataset/DRIVE/training/1st_manual/21_manual1.gif'))) > 0;
G = single(G);
Gb = logical(G);
fprintf('D2-focal GT vessel-frac=%.5f (gamma=2, NO alpha, clamp 1e-6, Dice x5)\n', mean(G(:)));
names = {'perfect','nothing','half','inverted','satwrong'};
Ps = {G, zeros(size(G),'single'), 0.5*ones(size(G),'single'), single(~Gb), single(~Gb)};
res = zeros(1, 5);
ok = true(1, 5);
for k = 1:5
    P = Ps{k};
    e = 1e-6;
    Pc = min(max(P, e), 1 - e);
    fo = -mean(G(:) .* (1-Pc(:)).^2 .* log(Pc(:)) + ...
               (1-G(:)) .* Pc(:).^2 .* log(1-Pc(:)));
    inter = sum(P(:) .* G(:));
    dice = (2*inter + 1) / (sum(P(:)) + sum(G(:)) + 1);
    di = 1 - dice;
    res(k) = fo + 5*di;
    ok(k) = isfinite(res(k));
    fprintf('D2-focal %-8s loss=%.4f (focal=%.4f di5=%.4f) finite=%d\n', ...
        names{k}, res(k), fo, 5*di, ok(k));
end
pass = all(ok) & (res(1) < res(3)) & (res(3) < res(2)) & (res(2) < res(4));
fprintf('D2-focal GATE perfect<half<nothing<inverted: %.4f<%.4f<%.4f<%.4f -> %s\n', ...
    res(1), res(3), res(2), res(4), string(pass));
fprintf('D2-focal half-vs-nothing margin=%.4f\n', res(2) - res(3));
if pass
    fprintf('D2-focal GATE PASS — training authorized.\n');
else
    fprintf('D2-focal GATE FAIL — STOP, no training.\n');
end
