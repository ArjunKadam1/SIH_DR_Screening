% dbgGate3 — reversed order + inspect connections
projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
S = load(fullfile(projRoot, 'results/vesselNet_weighted_5epoch.mat'), 'vesselNetWeightedTrained');
lg = layerGraph(S.vesselNetWeightedTrained);
cropL = 'encoderDecoderSkipConnectionCrop3';
gateL = 'Decoder-Stage-1-UpReLU';
mergeL = 'encoderDecoderSkipConnectionFeatureMerge3';
lg = disconnectLayers(lg, cropL, [mergeL '/in1']);
lg = addLayers(lg, [
    convolution2dLayer(1, 32, 'Name', 'AG1_Wx')
    convolution2dLayer(1, 32, 'Name', 'AG1_Wg')]);
lg = connectLayers(lg, gateL, 'AG1_Wg');
fprintf('gate->Wg FIRST ok\n');
disp(lg.Connections(end-2:end, :));
lg = connectLayers(lg, cropL, 'AG1_Wx');
fprintf('crop->Wx SECOND ok\n');
disp(lg.Connections(end-2:end, :));
