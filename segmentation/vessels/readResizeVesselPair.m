function dataOut = readResizeVesselPair(imageFile, maskFile)

    % Read fundus image
    I = imread(imageFile);

    % Read raw DRIVE vessel mask
    rawMask = imread(maskFile);

    % Convert 0/255 mask to binary
    vesselMask = rawMask == 255;

    % Resize image
    I = imresize(I, [256 256]);

    % Resize mask using nearest-neighbor
    vesselMask = imresize(vesselMask, [256 256], 'nearest');

    % Create categorical segmentation mask
    C = categorical( ...
        vesselMask, ...
        [false true], ...
        {'background','vessel'});

    % Return image + mask
    dataOut = {I, C};

end