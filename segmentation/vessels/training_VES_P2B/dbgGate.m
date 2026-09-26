% dbgGate — diagnose AG wiring port names (temporary debug, read-only + console only)
projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
S = load(fullfile(projRoot, 'results/vesselNet_weighted_5epoch.mat'), 'vesselNetWeightedTrained');
lg = layerGraph(S.vesselNetWeightedTrained);
lg = disconnectLayers(lg, 'encoderDecoderSkipConnectionCrop3', ...
    'encoderDecoderSkipConnectionFeatureMerge3/in1');
lg = addLayers(lg, convolution2dLayer(1, 32, 'Name', 'T_Wx'));
fprintf('added ok\n');
idx = find(string({lg.Layers.Name}) == "T_Wx");
L = lg.Layers(idx);
disp(L);
fprintf('--- connect crop -> T_Wx ---\n');
lg = connectLayers(lg, 'encoderDecoderSkipConnectionCrop3', 'T_Wx');
fprintf('crop-ok\n');
