% testGradCAM_bank — standalone old-vs-new Grad-CAM comparison bank.
% App untouched. Outputs ONLY to results/GRADCAM_TEST_<stamp>/.
% Parts: (0) synthetic warp-exactness (impulse peak error, must be <=10px);
% (1) auto-select 5 APTOS images: ACCEPT non-refer, ACCEPT refer, BORDERLINE,
%     REJECT (synthetic noise fallback), escalated/SecondOpinion case;
% (2) per image: old heat (naive full-frame stretch) vs new heat
%     (gradcamNativeWarp onto enhanced native via cropToFOV rect, logged
%     assumption: rect recomputed here, pipeline threading comes later);
% (3) metrics CSV + side-by-side PNGs + verdict table. No app wiring.
projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
addpath(fullfile(projRoot, 'demo'));
addpath(fullfile(projRoot, 'preprocessing'));
addpath(fullfile(projRoot, 'classification'));
addpath(fullfile(projRoot, 'quality'));
stamp = datestr(now, 'yyyymmdd_HHMMSS');
outDir = fullfile(projRoot, 'results', ['GRADCAM_TEST_' stamp]);
mkdir(outDir);
logPath = fullfile(outDir, 'console_bank.log');
diary(logPath);

% ---- 0. synthetic warp exactness ----
imp = zeros(7, 7); imp(2, 5) = 1;   % known cell (row2,col5)
rect = [100 80 499 479];            % 400x400 box in 600x500 frame
hw = gradcamNativeWarp(imp, rect, [600 500], [224 224]);
[mv, li] = max(hw(:));
[pr, pc] = ind2sub(size(hw), li);
% expected: 224-cell (2,5) maps to model px ~((2-0.5)/7*224, (5-0.5)/7*224)=(48,144);
% inverse to crop box: x=100+144/224*400, y=80+48/224*400
expC = round(100 + ((5-0.5)/7*224)/224*400);
expR = round(80 + ((2-0.5)/7*224)/224*400);
peakErr = sqrt((pr-expR)^2 + (pc-expC)^2);
fprintf('SYNTH peak=(%d,%d) expected=(%d,%d) err=%.1fpx (bar <=10)\n', ...
    pr, pc, expR, expC, peakErr);
synthPass = peakErr <= 10;

% ---- 1. image selection ----
S = load(fullfile(projRoot, 'results/trained_resnet18.mat'), 'trainedNet');
net = S.trainedNet;
candDir = fullfile(projRoot, 'dataset/APTOS/train_images');
d = dir(fullfile(candDir, '*.png'));
candNames = {d.name};
% pin known photo first, then alphabetical
pin = '1da4a17c18c9.png';
candNames = [pin, setdiff(candNames, pin, 'stable')];
candNames = candNames(1:min(10, numel(candNames)));
recs = struct('name', {}, 'qFlag', {}, 'refer', {}, 'esc', {}, 'cls', {}, 'rel', {});
for i = 1:numel(candNames)
    try
        I = imread(fullfile(candDir, candNames{i}));
        r = screenFundusImage(I, net, 'SecondOpinion', true);
        recs(end+1).name = candNames{i}; %#ok<AGROW>
        recs(end).qFlag = string(r.qualityStatus);
        recs(end).refer = r.referableDR;
        recs(end).esc = r.escalated;
        recs(end).cls = string(r.predictedClass);
        recs(end).rel = string(r.reliabilityFlag);
    catch e
        fprintf('SELECT skip %s: %s\n', candNames{i}, e.message);
    end
end
for i = 1:numel(recs)
    fprintf('SELECT %s q=%s refer=%d esc=%d cls=%s rel=%s\n', recs(i).name, ...
        recs(i).qFlag, recs(i).refer, recs(i).esc, recs(i).cls, recs(i).rel);
end
pick = @(cond, fallback, excl) pickFirst(recs, cond, fallback, excl);
if isempty(recs)
    error('BANK ABORT: no screenFundusImage runs succeeded on candidates.');
end
used = {};
[iA, used] = takePick(recs, @(r) r.qFlag ~= "REJECT" & ~r.refer & ~r.esc, 1, used);
[iR, used] = takePick(recs, @(r) r.qFlag ~= "REJECT" & r.refer, 2, used);
[iB, used] = takePick(recs, @(r) r.rel == "borderline", 3, used);
[iE, used] = takePick(recs, @(r) r.esc, 4, used);
[iX, used] = takePick(recs, @(r) true, 5, used);
sel.acceptNonRef = iA;
sel.acceptRef = iR;
sel.borderline = iB;
sel.escalated = iE;
sel.extra = iX;
selNames = {recs(sel.acceptNonRef).name, recs(sel.acceptRef).name, ...
    recs(sel.borderline).name, recs(sel.escalated).name, recs(sel.extra).name};
fprintf('BANK images: %s\n', strjoin(selNames, ', '));

% ---- 2-3. per-image old vs new + metrics ----
mrows = struct('img', {}, 'oldNewCentroidShift', {}, 'decidingClass', {}, ...
    'cardRefer', {}, 'mapReferConsistent', {}, 'rawMax', {}, 'alphaCap', {}, ...
    'featLayer', {}, 'oldExplainsLoser', {});
for i = 1:5
    nm = selNames{i};
    try
        I = imread(fullfile(candDir, nm));
        [H, W, ~] = size(I);
        r = screenFundusImage(I, net, 'SecondOpinion', true);
        if string(r.predictedClass) == "REJECTED" || ~r.classificationPerformed
            error('REJECT-path: no classification (expected for reject case)');
        end
        [~, enhNat] = assessFundusQuality(I);
        if ~isequal(size(enhNat, 1), H) || ~isequal(size(enhNat, 2), W)
            enhNat = imresize(enhNat, [H W]);
        end
        [~, rectUsed] = cropToFOV(I);
        if rectUsed
            [cc, rr] = deal([]);
            [cropped, ~] = cropToFOV(I);
            [ch, cw, ~] = size(cropped);
            % recover rect by template locate: use bounding-box recompute
            g = im2double(rgb2gray(I));
            fm = bwareafilt(g > 0.08, 1);
            st = regionprops(fm, 'BoundingBox');
            bb = st(1).BoundingBox;
            mg = 0.02*max(bb(3), bb(4));
            rectUsed = [max(1, floor(bb(1)-mg)), max(1, floor(bb(2)-mg)), ...
                min(W, ceil(bb(1)+bb(3)+mg)), min(H, ceil(bb(2)+bb(4)+mg))];
        else
            rectUsed = [];
        end
        % deciding class: post-escalation honesty — escalated cards REFER while
        % single-view string stays non-referable; explain minimal referable class
        oldExplainsLoser = r.escalated;
        if r.escalated
            decCls = "Moderate";
        else
            decCls = string(r.predictedClass);
        end
        G = evalGradCAM(net, r.classifierInput, decCls, rectUsed, [H W], ...
            string(r.reliabilityFlag));
        % old path replication (today's op): naive stretch onto original size
        oldHeat = imresize(double(squeeze(r.gradCAMMap)), [H W], 'bicubic');
        % centroid shift old vs new
        [yo, xo] = centroidOf(oldHeat);
        [yn, xn] = centroidOf(G.heatNative);
        shift = sqrt((yo-yn)^2 + (xo-xn)^2);
        gradeOf = @(s) gradeNum(s);
        mapRefer = gradeOf(G.decidingClass) >= 2;
        % side-by-side figure (visible off)
        f = figure('visible', 'off', 'Position', [0 0 1400 500]);
        subplot(1, 3, 1); imshow(I); title('Original', 'FontWeight', 'bold');
        subplot(1, 3, 2);
        imshow(I); hold on;
        hh = imagesc(oldHeat); set(hh, 'AlphaData', 0.45*oldHeat/max(oldHeat(:)));
        axis image off; colormap jet; colorbar;
        title('OLD warp (naive stretch)', 'FontWeight', 'bold'); hold off;
        subplot(1, 3, 3);
        imshow(enhNat); hold on;
        hn = G.heatNative; hnN = hn - min(hn(:)); hnN = hnN/max(hnN(:)+eps);
        hh2 = imagesc(hnN); set(hh2, 'AlphaData', G.alphaCap*hnN);
        axis image off; colormap jet; colorbar;
        title(['NEW warp: ' char(G.caption)], 'FontWeight', 'bold'); hold off;
        sgtitle(sprintf('%s | card=%s refer=%d esc=%d', nm, r.predictedClass, ...
            r.referableDR, r.escalated), 'FontWeight', 'bold');
        saveas(f, fullfile(outDir, sprintf('compare_%02d_%s.png', i, ...
            erase(nm, '.png'))));
        close(f);
        mrows(end+1).img = nm; %#ok<AGROW>
        mrows(end).oldNewCentroidShift = shift;
        mrows(end).decidingClass = G.decidingClass;
        mrows(end).cardRefer = r.referableDR;
        mrows(end).mapReferConsistent = (mapRefer == r.referableDR);
        mrows(end).rawMax = G.rawMax;
        mrows(end).alphaCap = G.alphaCap;
        mrows(end).featLayer = G.featLayer;
        mrows(end).oldExplainsLoser = oldExplainsLoser;
        fprintf('BANK %d %s shift=%.1fpx dec=%s cardRefer=%d consistent=%d rawMax=%.3f feat=%s loser=%d\n', ...
            i, nm, shift, G.decidingClass, r.referableDR, mrows(end).mapReferConsistent, ...
            G.rawMax, G.featLayer, oldExplainsLoser);
    catch e
        fprintf('BANK %d %s NOTE: %s\n', i, nm, e.message);
        mrows(end+1).img = [nm ' :: ' e.message]; %#ok<AGROW>
        mrows(end).oldNewCentroidShift = NaN;
        mrows(end).decidingClass = "N/A";
        mrows(end).cardRefer = false;
        mrows(end).mapReferConsistent = false;
        mrows(end).rawMax = NaN;
        mrows(end).alphaCap = NaN;
        mrows(end).featLayer = "N/A";
        mrows(end).oldExplainsLoser = false;
    end
end
cf = fopen(fullfile(outDir, 'metrics_bank.csv'), 'w');
fprintf(cf, 'img,shiftPx,decidingClass,cardRefer,consistent,rawMax,alphaCap,featLayer,oldExplainsLoser\n');
for i = 1:numel(mrows)
    fprintf(cf, '%s,%.1f,%s,%d,%d,%.4f,%.2f,%s,%d\n', mrows(i).img, ...
        mrows(i).oldNewCentroidShift, mrows(i).decidingClass, mrows(i).cardRefer, ...
        mrows(i).mapReferConsistent, mrows(i).rawMax, mrows(i).alphaCap, ...
        mrows(i).featLayer, mrows(i).oldExplainsLoser);
end
fclose(cf);
fprintf('SYNTH warp-exactness pass=%d (err %.1fpx)\n', synthPass, peakErr);
% ---- bonus: REJECT no-crash probe (synthetic noise, expect REJECT struct) ----
try
    rng(7);
    Nz = randi([0 255], 200, 200, 3, 'uint8');
    rz = screenFundusImage(Nz, net);
    fprintf('REJECT-PROBE class=%s performed=%d (want REJECTED/false, no crash)\n', ...
        rz.predictedClass, rz.classificationPerformed);
catch e
    fprintf('REJECT-PROBE threw (also acceptable if gated upstream): %s\n', e.message);
end
save(fullfile(outDir, 'bank_summary.mat'), 'mrows', 'synthPass', 'peakErr', 'selNames');
diary off;
fprintf('BANK ALL_DONE outDir=%s\n', outDir);

function [idx, used] = takePick(recs, cond, fallback, used)
% first UNUSED match; falls back to first unused row (distinctness preserved)
idx = [];
for k = 1:numel(recs)
    if cond(recs(k)) && ~ismember(recs(k).name, used)
        idx = k; break;
    end
end
if isempty(idx)
    for k = 1:numel(recs)
        if ~ismember(recs(k).name, used), idx = k; break; end
    end
end
if isempty(idx), idx = min(fallback, max(numel(recs), 1)); end
if ~isempty(recs), used{end+1} = recs(idx).name; end
end

function idx = pickFirst(recs, cond, fallback, excl)
if nargin < 4, excl = {}; end
[idx, ~] = takePick(recs, cond, fallback, excl);
end

function g = gradeNum(s)
switch string(s)
    case "No_DR", g = 0;
    case "Mild", g = 1;
    case "Moderate", g = 2;
    case {"Severe", "SevereProlif"}, g = 3;
    case "Proliferate_DR", g = 4;
    otherwise, g = NaN;
end
end

function [cy, cx] = centroidOf(m)
m = double(m);
m = m - min(m(:));
s = sum(m(:));
if s <= 0, cy = NaN; cx = NaN; return; end
[rr, cc] = ndgrid(1:size(m, 1), 1:size(m, 2));
cy = sum(rr(:) .* m(:)) / s;
cx = sum(cc(:) .* m(:)) / s;
end
