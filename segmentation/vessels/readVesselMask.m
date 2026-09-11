function mask = readVesselMask(filename)

mask = imread(filename);

% Convert 0/255 mask to logical binary mask
mask = mask > 0;

end