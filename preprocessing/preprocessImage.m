function enhancedRGB = preprocessImage(img)

    % Convert RGB to Lab color space
    labImg = rgb2lab(img);

    % Extract luminance channel
    L = labImg(:,:,1);

    % Normalize luminance to [0,1]
    L = L / 100;

    % Apply CLAHE
    L_enhanced = adapthisteq(L, ...
        'ClipLimit', 0.005, ...
        'NumTiles', [8 8]);

    % Put enhanced luminance back
    labImg(:,:,1) = L_enhanced * 100;

    % Convert back to RGB
    enhancedRGB = lab2rgb(labImg);

    % Return uint8 RGB image
    enhancedRGB = im2uint8(enhancedRGB);

end