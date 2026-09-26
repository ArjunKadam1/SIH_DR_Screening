function heatNative = gradcamNativeWarp(mapModel, cropRect, frameSize, modelSize)
% GRADCAMNATIVEWARP — pure exact-inverse warp, no model/app dependency.
% mapModel : gradCAM map in MODEL space (e.g. 7x7 from gradCAM, any 2D).
% cropRect : [x1 y1 x2 y2] full-frame pixels of the bbox crop fed to the
%            classifier ([] = crop failed closed -> direct warp to frame).
% frameSize: [H W] full-frame (original) image size.
% modelSize: [mh mw] classifier input size (e.g. [224 224]).
% Chain inverted exactly: map -> modelSize (bicubic, today's op) ->
% crop size with EXACT inverse factors -> embed at (x1,y1) in zero canvas.
% Non-square crops keep non-square heat pixels (logged limitation, not fixed:
% letterboxing model input would change verdicts — out of scope).
mapModel = double(squeeze(mapModel));
mh = modelSize(1); mw = modelSize(2);
H = frameSize(1); W = frameSize(2);
map224 = imresize(mapModel, [mh mw], 'bicubic');
if isempty(cropRect)
    heatNative = imresize(map224, [H W], 'bicubic');
    return;
end
x1 = cropRect(1); y1 = cropRect(2);
x2 = cropRect(3); y2 = cropRect(4);
cw = x2 - x1 + 1; ch = y2 - y1 + 1;
mapCrop = imresize(map224, [ch cw], 'bicubic');
heatNative = zeros(H, W);
x1c = max(1, x1); y1c = max(1, y1);
x2c = min(W, x2); y2c = min(H, y2);
sx1 = x1c - x1 + 1; sy1 = y1c - y1 + 1;
sx2 = sx1 + (x2c - x1c); sy2 = sy1 + (y2c - y1c);
heatNative(y1c:y2c, x1c:x2c) = mapCrop(sy1:sy2, sx1:sx2);
end
