function [contrastValue, qualityFlag] = qualityCheck(img)

% Convert RGB image to grayscale
grayImg = rgb2gray(img);

% Calculate image contrast
contrastValue = std(double(grayImg(:)));

% Quality threshold obtained from dataset sample
qualityThreshold = 11.45;

% Quality decision
if contrastValue >= qualityThreshold
    qualityFlag = "Good";
else
    qualityFlag = "Low Contrast";
end

end