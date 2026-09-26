% dbgGate2 — replicate build-script gate-1 sequence exactly
projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
S = load(fullfile(projRoot, 'results/vesselNet_weighted_5epoch.mat'), 'vesselNetWeightedTrained');
lg = layerGraph(S.vesselNetWeightedTrained);
cropL = 'encoderDecoderSkipConnectionCrop3';
gateL = 'Decoder-Stage-1-UpReLU';
mergeL = 'encoderDecoderSkipConnectionFeatureMerge3';
Cx = 64; Cg = 64; Fi = 32; p = 'AG1';
lg = disconnectLayers(lg, cropL, [mergeL '/in1']);
fprintf('disconnect ok\n');
lg = addLayers(lg, [
    convolution2dLayer(1, Fi, 'Name', [p '_Wx'])
    convolution2dLayer(1, Fi, 'Name', [p '_Wg'])
    additionLayer(2, 'Name', [p '_Add'])
    reluLayer('Name', [p '_ReLU'])
    convolution2dLayer(1, 1, 'Name', [p '_Psi'])
    sigmoidLayer('Name', [p '_Sig'])
    multiplicationLayer(2, 'Name', [p '_Mult'])]);
fprintf('add ok\n');
lg = connectLayers(lg, cropL, [p '_Wx']);
fprintf('crop->Wx ok\n');
lg = connectLayers(lg, gateL, [p '_Wg']);
fprintf('gate->Wg ok\n');
