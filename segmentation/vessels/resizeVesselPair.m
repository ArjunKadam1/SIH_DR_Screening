function [I, C] = resizeVesselPair(I, C)

    % Resize fundus image
    I = imresize(I, [256 256]);

    % Convert categorical mask to binary
    vesselMask = (C == 'vessel');

    % Resize binary vessel mask
    vesselMask = imresize(vesselMask, [256 256], 'nearest');

    % Create background mask
    C = categorical(repmat("background", 256, 256));

    % Put vessel pixels back
    C(vesselMask) = categorical("vessel");

end