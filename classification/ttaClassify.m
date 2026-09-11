function [label, meanScores, allScores] = ttaClassify(enhancedImage, net, nViews)
%% TTACLASSIFY - test-time multi-scale ensemble for scale robustness
%
% Classifies N deterministic views (5-crop at one scale, or 5-crop at two
% scales for 10 views) and averages softmax. Makes one batched classify
% call, so cost is seconds on GPU.
%
% Views are deterministic (corners + center, no randomness) so the demo is
% reproducible: same image -> same prediction, every run.
%
% Inputs:
%   enhancedImage - full-resolution enhanced RGB (NOT pre-shrunk to 224;
%                   pass the full-size image so crops carry real content)
%   net           - trained DAGNetwork (ResNet-18 5-class)
%   nViews        - 5 (default, one scale) or 10 (two scales)
%
% Outputs:
%   label       - categorical predicted class (mean-softmax winner)
%   meanScores  - 1x5 mean softmax vector (same class order as classify)
%   allScores   - Nx5 per-view softmax matrix

if nargin < 3 || isempty(nViews)
    nViews = 5;
end

if size(enhancedImage,3) == 1
    enhancedImage = repmat(enhancedImage,1,1,3);
end
enhancedImage = im2uint8(enhancedImage);

views = buildTTAViews(enhancedImage, nViews); % 224x224x3xN
[~, allScores] = classify(net, views);
meanScores = mean(allScores, 1);

try
    classes = net.Layers(end).Classes;
catch
    [p1, ~] = classify(net, views(:,:,:,1));
    classes = categories(p1);
end
[~, bi] = max(meanScores);
label = classes(bi);
end

function views = buildTTAViews(img, nViews)
%% 5-crop at short-side 256; second scale at 352 if 10 views requested
[H, W, ~] = size(img);
shortSide = min(H, W);

scales = 256 / shortSide;
if nViews >= 10
    scales = [256, 352] / shortSide;
end

views = zeros(224,224,3,nViews,'uint8');
nFilled = 0;
for s = 1:numel(scales)
    Rs = max(round(H*scales(s)), 224);
    Cs = max(round(W*scales(s)), 224);
    R = imresize(img, [Rs Cs]);
    rows = [1, Rs-224+1, floor((Rs-224)/2)+1];
    cols = [1, Cs-224+1, floor((Cs-224)/2)+1];
    crops = { ...
        R(rows(1):rows(1)+223, cols(1):cols(1)+223, :), ...
        R(rows(1):rows(1)+223, cols(2):cols(2)+223, :), ...
        R(rows(2):rows(2)+223, cols(1):cols(1)+223, :), ...
        R(rows(2):rows(2)+223, cols(2):cols(2)+223, :), ...
        R(rows(3):rows(3)+223, cols(3):cols(3)+223, :)};
    for c = 1:5
        if nFilled >= nViews
            break;
        end
        nFilled = nFilled + 1;
        views(:,:,:,nFilled) = crops{c};
    end
    if nFilled >= nViews
        break;
    end
end
views = views(:,:,:,1:nFilled);
end
