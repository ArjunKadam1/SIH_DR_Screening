% buildVesselAttentionP7 — P7: graft P6-style 3 attention gates onto the
% P3-A BEST net (armA_block2_iter80, pooled dFix 0.3735, resize protocol).
% Same graft as P6 (Fint=Cx/2, individual addLayers, 69 layers total).
% Warm-start: copy ALL matching learnables from the P3-A net (trained 80
% resize iters past plain init); fresh only the 18 new AG tensors.
% Expectation logged: fresh sigmoid gates output ~0.5, halving skip signals,
% so initial Dice will DIP below 0.37 until gates learn — trajectory matters.
% D1: fixed-batch forward shape + grad reach all gates. Saves attInitP7.mat.
% New files only; P3-A + P6 dirs untouched.

projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
addpath(fullfile(projRoot, 'segmentation/vessels/training_VES_P2B'));
outDir = fullfile(projRoot, 'results/VES_P7_ATTRESIZE_20260923');
if ~isfolder(outDir), mkdir(outDir); end
logPath = fullfile(outDir, 'build_attentionP7.log');
diary(logPath);

S = load(fullfile(projRoot, 'results/VES_P3_NORM_20260923/arm_resize/armA_block2_iter80.mat'), 'net');
plain = S.net;
fprintf('P7-BASE net=P3-A armA_block2 (pooled dFix 0.3735, 80 resize iters)\n');
lg = layerGraph(plain);

% skip definitions: {cropLayer, gatingUpReLU, mergeLayer, Cx, Cg, Fint}
G = { ...
    'encoderDecoderSkipConnectionCrop3', 'Decoder-Stage-1-UpReLU', ...
    'encoderDecoderSkipConnectionFeatureMerge3', 64, 64, 32; ...
    'encoderDecoderSkipConnectionCrop2', 'Decoder-Stage-2-UpReLU', ...
    'encoderDecoderSkipConnectionFeatureMerge2', 32, 32, 16; ...
    'encoderDecoderSkipConnectionCrop1', 'Decoder-Stage-3-UpReLU', ...
    'encoderDecoderSkipConnectionFeatureMerge1', 16, 16, 8};

for k = 1:3
    cropL = G{k,1}; gateL = G{k,2}; mergeL = G{k,3};
    Cx = G{k,4}; Cg = G{k,5}; Fi = G{k,6};
    p = sprintf('AG%d', k);
    lg = disconnectLayers(lg, cropL, [mergeL '/in1']);
    % NOTE: addLayers with an ARRAY auto-connects layers in sequence, which
    % would chain Wx->Wg and steal Wg's input. Add one at a time instead.
    lg = addLayers(lg, convolution2dLayer(1, Fi, 'Name', [p '_Wx']));
    lg = addLayers(lg, convolution2dLayer(1, Fi, 'Name', [p '_Wg']));
    lg = addLayers(lg, additionLayer(2, 'Name', [p '_Add']));
    lg = addLayers(lg, reluLayer('Name', [p '_ReLU']));
    lg = addLayers(lg, convolution2dLayer(1, 1, 'Name', [p '_Psi']));
    lg = addLayers(lg, sigmoidLayer('Name', [p '_Sig']));
    lg = addLayers(lg, multiplicationLayer(2, 'Name', [p '_Mult']));
    lg = connectLayers(lg, cropL, [p '_Wx']);
    lg = connectLayers(lg, gateL, [p '_Wg']);
    lg = connectLayers(lg, [p '_Wx'], [p '_Add/in1']);
    lg = connectLayers(lg, [p '_Wg'], [p '_Add/in2']);
    lg = connectLayers(lg, [p '_Add'], [p '_ReLU']);
    lg = connectLayers(lg, [p '_ReLU'], [p '_Psi']);
    lg = connectLayers(lg, [p '_Psi'], [p '_Sig']);
    lg = connectLayers(lg, cropL, [p '_Mult/in1']);
    lg = connectLayers(lg, [p '_Sig'], [p '_Mult/in2']);
    lg = connectLayers(lg, [p '_Mult'], [mergeL '/in1']);
    fprintf('GATE %s: skip=%s(%dch) gating=%s(%dch) Fint=%d\n', p, cropL, Cx, gateL, Cg, Fi);
end

attNet = dlnetwork(lg);
fprintf('ASSEMBLED layers=%d (plain 48 + 21 gate layers)\n', numel(attNet.Layers));

% --- weight map: copy by (Layer,Parameter) name; new AG tensors keep init ---
pT = plain.Learnables; aT = attNet.Learnables;
nCopy = 0; nNew = 0; newList = {};
for i = 1:height(aT)
    ln = string(aT.Layer(i)); pn = string(aT.Parameter(i));
    hit = find(string(pT.Layer) == ln & string(pT.Parameter) == pn, 1);
    if ~isempty(hit)
        attNet.Learnables.Value{i} = pT.Value{hit};
        nCopy = nCopy + 1;
    else
        nNew = nNew + 1;
        newList{end+1} = char(ln + "/" + pn); %#ok<AGROW>
    end
end
nPrm = 0; nPrmNew = 0;
for i = 1:height(aT)
    n = numel(gather(extractdata(aT.Value{i})));
    nPrm = nPrm + n;
    ln = string(aT.Layer(i));
    if startsWith(ln, "AG"), nPrmNew = nPrmNew + n; end
end
fprintf('MAP copied-tensors=%d new-tensors=%d new-params=%d/%d (%.2f%%)\n', ...
    nCopy, nNew, nPrmNew, nPrm, 100*nPrmNew/nPrm);
fprintf('MAP new list:\n'); fprintf('  %s\n', newList{:});
assert(nNew == 18, 'Expected 18 new AG tensors (3 gates x 6) — STOP.');
fp = 0;
for i = 1:height(aT)
    v = double(gather(extractdata(attNet.Learnables.Value{i})));
    fp = fp + sum(v(:).^2);
end
fprintf('INIT fpNorm=%.6f\n', sqrt(fp));
save(fullfile(outDir, 'attInitP7.mat'), 'attNet', 'nCopy', 'nNew');

% --- D1 shape + grad-reach check on fixed batch (4 native patches, img 21) ---
I21 = imread(fullfile(projRoot, 'dataset/DRIVE/training/images/21_training.tif'));
V21 = imread(fullfile(projRoot, 'dataset/DRIVE/training/1st_manual/21_manual1.gif')) > 0;
ps = 256;
Xb = zeros(ps, ps, 3, 4, 'single'); Tb = zeros(ps, ps, 2, 4, 'single');
coords = [1 1; 1 129; 129 1; 129 129];
for j = 1:4
    r = coords(j,1); c = coords(j,2);
    Xb(:,:,:,j) = im2single(I21(r:r+ps-1, c:c+ps-1, :));
    Vp = V21(r:r+ps-1, c:c+ps-1);
    Tb(:,:,1,j) = single(~Vp); Tb(:,:,2,j) = single(Vp);
end
useGPU = (gpuDeviceCount > 0);
if useGPU, Xd = gpuArray(dlarray(Xb,'SSCB')); Td = gpuArray(dlarray(Tb,'SSCB'));
else, Xd = dlarray(Xb,'SSCB'); Td = dlarray(Tb,'SSCB'); end
Y = forward(attNet, Xd);
sz = size(gather(extractdata(Y)));
fprintf('D1 forward size=%s (expect [256 256 2 4])\n', mat2str(sz));
assert(isequal(sz, [256 256 2 4]), 'D1 shape mismatch — STOP.');
[lv, gr, foP, diP] = dlfeval(@vesselLoss_focal, attNet, Xd, Td);
fprintf('D1 loss=%.4f (focal=%.3f di=%.3f)\n', gather(extractdata(lv)), ...
    gather(extractdata(foP)), gather(extractdata(diP)));
gT = gr; allFin = true; agHit = 0; agNZ = 0;
for gi = 1:height(gT)
    gv = gather(extractdata(gT.Value{gi}));
    allFin = allFin && all(isfinite(gv(:)));
    if startsWith(string(gT.Layer(gi)), "AG")
        agHit = agHit + 1;
        if any(gv(:) ~= 0), agNZ = agNZ + 1; end
    end
end
fprintf('D1 grads finite=%d AG-tensors-with-grad=%d/%d nonzero=%d\n', ...
    allFin, agHit, 18, agNZ);
assert(allFin && agHit == 18, 'D1 grad-reach failed — STOP.');
fprintf('D1 PASS — attention net built, warm-started, gradients reach all gates.\n');
diary off;
