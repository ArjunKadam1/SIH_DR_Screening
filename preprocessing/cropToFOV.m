function [cropped, usedCrop] = cropToFOV(img)
%% CROPTOFOV - crop to the fundus field-of-view disc, distortion-free input
%
% Removes black borders / vignette surround so the later resize to 224x224
% stretches actual retina instead of squishing frame + borders (which also
% distorts aspect ratio on portrait images like 1958x2588).
%
% Fails closed: any suspicious mask returns the ORIGINAL image with
% usedCrop=false, so behavior degrades to today's squish, never worse.
% (Threshold-FOV is known-unreliable on some dataset backgrounds, hence
% the sanity gate below.)
%
% Usage:
%   [cropped, usedCrop] = cropToFOV(img)

usedCrop = false;
cropped = img;

try
    grayDouble = im2double(rgb2gray(img));

    % Same mask recipe as qualityCheckV2 (threshold + largest component)
    fovMask = grayDouble > 0.08;
    fovMask = bwareafilt(fovMask, 1);

    coverage = sum(fovMask(:)) / numel(fovMask);
    if coverage < 0.05 || coverage > 0.95
        return; % degenerate mask -> fall back
    end

    stats = regionprops(fovMask, 'BoundingBox');
    if isempty(stats)
        return;
    end
    bb = stats(1).BoundingBox; % [x y w h]

    % Reject masks touching >1 frame edge (leaked background)
    H = size(img,1); W = size(img,2);
    touches = (bb(1) <= 1) + (bb(2) <= 1) + ...
              (bb(1)+bb(3) >= W) + (bb(2)+bb(4) >= H);
    if touches > 1
        return;
    end

    margin = 0.02 * max(bb(3), bb(4));
    x1 = max(1, floor(bb(1)-margin));
    y1 = max(1, floor(bb(2)-margin));
    x2 = min(W, ceil(bb(1)+bb(3)+margin));
    y2 = min(H, ceil(bb(2)+bb(4)+margin));
    if (x2-x1) < 50 || (y2-y1) < 50
        return; % absurdly small -> fall back
    end

    cropped = img(y1:y2, x1:x2, :);
    usedCrop = true;
catch
    % Any failure -> original image (fails closed)
    cropped = img;
    usedCrop = false;
end
end
