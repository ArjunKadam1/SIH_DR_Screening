function predMask = predictMAImageTiled(I, maNet)
% predictMAImageTiled
% Runs the 512x512 MA U-Net over a high-resolution fundus image
% using overlapping tiles and stitches the predictions together.

patchSize = 512;
stride = 256;

I = im2uint8(I);

[H,W,~] = size(I);

% ---------------------------------------------------------
% Pad image so every region is covered
% ---------------------------------------------------------

if H < patchSize
    Hpad = patchSize;
else
    Hpad = H;
end

if W < patchSize
    Wpad = patchSize;
else
    Wpad = W;
end

% Ensure final tile reaches image boundary
Hpad = Hpad + mod(stride - mod(Hpad-patchSize,stride),stride);
Wpad = Wpad + mod(stride - mod(Wpad-patchSize,stride),stride);

Ipad = zeros(Hpad,Wpad,3,'uint8');
Ipad(1:H,1:W,:) = I;

% ---------------------------------------------------------
% Accumulator for MA predictions
% ---------------------------------------------------------

predictionSum = zeros(Hpad,Wpad,'double');
predictionCount = zeros(Hpad,Wpad,'double');

rowStarts = 1:stride:(Hpad-patchSize+1);
colStarts = 1:stride:(Wpad-patchSize+1);

fprintf('\nRunning tiled MA inference...\n');

totalTiles = numel(rowStarts)*numel(colStarts);
tileNumber = 0;

for r = rowStarts

    for c = colStarts

        tileNumber = tileNumber + 1;

        tile = Ipad(r:r+patchSize-1, ...
            c:c+patchSize-1,:);

        % Run segmentation
        tilePred = semanticseg(tile,maNet);

        classNames = categories(tilePred);

        % Second class = microaneurysm
        maTile = tilePred == classNames{2};

        % Accumulate overlapping predictions
        predictionSum(r:r+patchSize-1, ...
            c:c+patchSize-1) = ...
            predictionSum(r:r+patchSize-1, ...
            c:c+patchSize-1) + double(maTile);

        predictionCount(r:r+patchSize-1, ...
            c:c+patchSize-1) = ...
            predictionCount(r:r+patchSize-1, ...
            c:c+patchSize-1) + 1;

        if mod(tileNumber,10) == 0 || tileNumber == totalTiles
            fprintf('Tiles: %d / %d\n', ...
                tileNumber,totalTiles);
        end
    end
end

% ---------------------------------------------------------
% Average overlapping predictions
% ---------------------------------------------------------

predictionProbability = ...
    predictionSum ./ max(predictionCount,1);

% Majority vote
predMask = predictionProbability >= 0.5;

% Crop back to original image
predMask = predMask(1:H,1:W);

end