function [I, C] = readMAPatch(entry)
% readMAPatch
% Loads one MA image patch and its corresponding segmentation mask
% from a saved MA batch file.

% Load the required batch
B = load(entry.batchFile,'batchInfo');

% Retrieve image and binary mask
if strcmp(entry.patchType,'positive')

    I = B.batchInfo.positivePatches{entry.patchNumber};
    M = B.batchInfo.positiveMasks{entry.patchNumber};

elseif strcmp(entry.patchType,'negative')

    I = B.batchInfo.negativePatches{entry.patchNumber};
    M = B.batchInfo.negativeMasks{entry.patchNumber};

else
    error('Unknown patch type: %s',entry.patchType);
end

% Ensure RGB uint8 image
I = im2uint8(I);

% Convert binary mask to categorical segmentation labels
C = categorical( ...
    M, ...
    [false true], ...
    ["background","microaneurysm"]);

end