function cachedImds = buildV2Cache(imdsIn, cacheDir, imageSize)
%% BUILDV2CACHE - one-time Lab+CLAHE preprocess cache for V2 training
%
% Same pixels as preprocessFromFile, just done ONCE to disk instead of
% re-running Lab+CLAHE every image every epoch (the CPU bottleneck).
%
% Inputs:
%   imdsIn    - imageDatastore with Files + Labels (e.g. balanced train)
%   cacheDir  - folder to write 224x224 PNGs, per-class subfolders
%   imageSize - default [224 224]
%
% Output:
%   cachedImds - imageDatastore pointing at cacheDir (Labels from folders,
%                ReadFcn = default imread since images are already enhanced)
%
% Honesty: split membership unchanged, only pixel precompute is cached.
% Replicas (oversampled) get unique _rNNNN names so counts match exactly.

if nargin < 3 || isempty(imageSize)
    imageSize = [224 224];
end

files = imdsIn.Files;
labels = imdsIn.Labels;
n = numel(files);
fprintf('buildV2Cache: %d images -> %s\n', n, cacheDir);

% Check for valid existing cache (same count, all files present)
if isfolder(cacheDir)
    existing = dir(fullfile(cacheDir, '*', '*.png'));
    if numel(existing) == n
        fprintf('buildV2Cache: cache hit (%d files), skipping rebuild.\n', n);
        cachedImds = imageDatastore(cacheDir, ...
            'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        return;
    else
        fprintf('buildV2Cache: cache has %d files, need %d - rebuilding.\n', ...
            numel(existing), n);
    end
end

if ~isfolder(cacheDir)
    mkdir(cacheDir);
end

% Write each preprocessed image to per-class subfolder with unique name
for i = 1:n
    if iscell(files)
        f = files{i};
    else
        f = char(files(i));
    end
    lab = char(string(labels(i)));
    classDir = fullfile(cacheDir, lab);
    if ~isfolder(classDir)
        mkdir(classDir);
    end
    [~, bn, ~] = fileparts(f);
    outName = sprintf('%s_r%04d.png', bn, i);
    outPath = fullfile(classDir, outName);
    if ~isfile(outPath)
        try
            img = preprocessFromFile(f); %#ok<NASGU> % Lab+CLAHE, full size
        catch e
            warning('buildV2Cache: preprocess failed for %s: %s', f, e.message);
            continue;
        end
        img224 = imresize(img, imageSize);
        imwrite(img224, outPath);
    end
    if mod(i, 500) == 0 || i == n
        fprintf('buildV2Cache: %d/%d cached.\n', i, n);
    end
end

cachedImds = imageDatastore(cacheDir, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
fprintf('buildV2Cache: done, %d cached images.\n', numel(cachedImds.Files));
end
