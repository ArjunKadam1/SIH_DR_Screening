function [qualityResult, enhancedImage] = assessFundusQuality(img)

% ============================================================
% FUNDUS IMAGE QUALITY ASSESSMENT + GENTLE ENHANCEMENT
% V2
% ============================================================

if size(img,3) ~= 3
    error('Input must be an RGB fundus image.');
end

img = im2uint8(img);

% ------------------------------------------------------------
% 1. Approximate retinal FOV mask
% ------------------------------------------------------------

gray = rgb2gray(img);

fovMask = gray > 0.08 * double(intmax(class(gray)));

fovMask = bwareaopen(fovMask, 500);
fovMask = bwareafilt(fovMask, 1);

% ------------------------------------------------------------
% 2. FOV coverage
% ------------------------------------------------------------

fovCoverage = nnz(fovMask) / numel(fovMask);

% ------------------------------------------------------------
% 3. Focus score
% ------------------------------------------------------------

grayDouble = im2double(gray);

[Gx, Gy] = imgradientxy(grayDouble);
gradientMagnitude = sqrt(Gx.^2 + Gy.^2);

focusScore = mean(gradientMagnitude(fovMask));

% ------------------------------------------------------------
% 4. Illumination
% ------------------------------------------------------------

retinalPixels = grayDouble(fovMask);

meanIllumination = mean(retinalPixels);
illuminationStd = std(retinalPixels);

% ------------------------------------------------------------
% 5. Quality decision
% ------------------------------------------------------------

% Engineering prototype thresholds.
% These are NOT clinical thresholds.

if focusScore < 0.0016

    qualityStatus = "REJECT";
    qualityReason = "Image appears too blurry.";

elseif meanIllumination < 0.2340

    qualityStatus = "BORDERLINE";
    qualityReason = "Image appears poorly illuminated.";

elseif fovCoverage < 0.475

    qualityStatus = "BORDERLINE";
    qualityReason = "Retinal field of view appears limited.";

else

    qualityStatus = "ACCEPT";
    qualityReason = "Image quality is acceptable.";

end

% ------------------------------------------------------------
% 6. GENTLE ENHANCEMENT
% ------------------------------------------------------------

if qualityStatus == "BORDERLINE"

    % Convert RGB to LAB
    labImg = rgb2lab(img);

    % Extract luminance channel
    L = labImg(:,:,1) / 100;

    % --------------------------------------------------------
    % Mild gamma correction
    % Helps dark images without aggressively stretching them.
    % --------------------------------------------------------

    L_gamma = imadjust(L, [], [], 0.85);

    % --------------------------------------------------------
    % Mild CLAHE
    % Lower ClipLimit prevents excessive texture amplification.
    % --------------------------------------------------------

    L_clahe = adapthisteq( ...
        L_gamma, ...
        'ClipLimit', 0.003, ...
        'NumTiles', [8 8]);

    % --------------------------------------------------------
    % Blend instead of replacing luminance completely.
    % This preserves the natural appearance of the fundus.
    % --------------------------------------------------------

    L_enhanced = ...
        0.65 * L_gamma + ...
        0.35 * L_clahe;

    % Keep intensity safely within range
    L_enhanced = min(max(L_enhanced, 0), 1);

    labImg(:,:,1) = L_enhanced * 100;

    enhancedImage = lab2rgb(labImg);

    % --------------------------------------------------------
    % Mild denoising
    % --------------------------------------------------------

    enhancedImage = imgaussfilt(enhancedImage, 0.4);

    enhancedImage = im2uint8(enhancedImage);

else

    % No enhancement required
    enhancedImage = img;

end

% ------------------------------------------------------------
% 7. Return quality information
% ------------------------------------------------------------

qualityResult.FocusScore = focusScore;
qualityResult.FOVCoverage = fovCoverage;
qualityResult.MeanIllumination = meanIllumination;
qualityResult.IlluminationStd = illuminationStd;
qualityResult.Status = qualityStatus;
qualityResult.Reason = qualityReason;

end