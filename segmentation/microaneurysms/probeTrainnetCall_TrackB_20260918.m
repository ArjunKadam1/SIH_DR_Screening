% probeTrainnetCall_TrackB_20260918 — discover trainnet loss-call convention.
projRoot = 'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening';
imdsFull = imageDatastore(fullfile(projRoot, 'segmentation/microaneurysms/patches/train/images'));
pxdsFull = pixelLabelDatastore(fullfile(projRoot, 'segmentation/microaneurysms/patches/train/masks'), ...
    ["background", "MA"], [0 1]);
imds = imageDatastore(imdsFull.Files(1:32));
pxds = pixelLabelDatastore(pxdsFull.Files(1:32), ["background", "MA"], [0 1]);
cds = transform(combine(imds, pxds), @plainRead);
dlnet = unet([256 256 3], 2);
opts = trainingOptions('adam', 'MaxEpochs', 1, 'MiniBatchSize', 16, ...
    'Plots', 'none', 'Verbose', false, 'ExecutionEnvironment', 'gpu');
net = trainnet(cds, dlnet, @probeLoss_TrackB_20260918, opts);

function out = plainRead(data)
out = {im2single(data{1}), cat(3, single(data{2} == 'background'), single(data{2} == 'MA'))};
end
