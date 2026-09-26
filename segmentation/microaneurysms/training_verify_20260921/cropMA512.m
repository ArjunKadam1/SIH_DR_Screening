function [Xb, Tb] = cropMA512(man, idx)
% cropMA512 — NEW helper (2026-09-21). Strict 512 live crops from manifest.
% IDRiD mask rule: M>0. eOphtha mask rule: M>=200 (verified vs LesionPixels).
% No padding/clamping: asserts X+511<=W, Y+511<=H.
% Xb: 512x512x3xN single (image), Tb: 512x512x2xN single (one-hot bg/MA).
N = numel(idx);
Xb = zeros(512, 512, 3, N, 'single');
Tb = zeros(512, 512, 2, N, 'single');
for j = 1:N
    r = man(idx(j), :);
    s = string(r.ImagePath);
    if contains(s, 'IDRiD')
        ipath = char(strrep(strrep(s, '/MATLAB Drive/SIH_DR_Screening/dataset/IDRiD/', ...
            'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening/dataset/IDRiD/'), '/', '\'));
        sp = string(r.MaskPath);
        mpath = char(strrep(strrep(sp, '/MATLAB Drive/SIH_DR_Screening/dataset/IDRiD/', ...
            'C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening/dataset/IDRiD/'), '/', '\'));
        I = imread(ipath); M = imread(mpath) > 0;
    else
        ipath = char(strrep(strrep(s, '/MATLAB Drive/SIH_DR_Screening/dataset/e_optha_MA/', ...
            'C:/Users/SHIVANYA SALES/Downloads/e_optha_MA/'), '/', '\'));
        sp = string(r.MaskPath);
        mpath = char(strrep(strrep(sp, '/MATLAB Drive/SIH_DR_Screening/dataset/e_optha_MA/', ...
            'C:/Users/SHIVANYA SALES/Downloads/e_optha_MA/'), '/', '\'));
        I = imread(ipath); M = imread(mpath) >= 200;
    end
    assert(r.X + 511 <= size(I, 2) && r.Y + 511 <= size(I, 1), 'Crop out of bounds — STOP.');
    Xb(:, :, :, j) = im2single(I(r.Y:r.Y+511, r.X:r.X+511, :));
    Mp = M(r.Y:r.Y+511, r.X:r.X+511);
    Tb(:, :, 1, j) = single(~Mp);
    Tb(:, :, 2, j) = single(Mp);
end
end
