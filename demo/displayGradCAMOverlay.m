function displayGradCAMOverlay(originalImage, enhancedImage, gradCAMMap, result)
% DISPLAYGRADCAMOVERLAY
% Clean visualization of actual fundus image + Grad-CAM.

if nargin < 4
    result = struct();
end

%% Prepare images
originalImage = im2uint8(originalImage);
enhancedImage = im2uint8(enhancedImage);

%% Resize Grad-CAM to original image dimensions
[H,W,~] = size(originalImage);
gradCAMFull = imresize(gradCAMMap,[H W]);

%% Normalize Grad-CAM
gradCAMFull = double(gradCAMFull);
gradCAMFull = gradCAMFull - min(gradCAMFull(:));

maxValue = max(gradCAMFull(:));

if maxValue > 0
    gradCAMFull = gradCAMFull ./ maxValue;
end

%% Create figure
figure('Name','SIH V2 - Fundus Grad-CAM Explanation', ...
    'Color','w');

%% Original
subplot(2,2,1);
imshow(originalImage);
title('Original Fundus','FontWeight','bold');

%% Enhanced
subplot(2,2,2);
imshow(enhancedImage);
title('V2 Enhanced Fundus','FontWeight','bold');

%% Grad-CAM only
subplot(2,2,3);
imagesc(gradCAMFull);
axis image off;
colormap jet;
colorbar;
title('Grad-CAM Attention','FontWeight','bold');

%% Actual retina + Grad-CAM
subplot(2,2,4);
imshow(originalImage);
hold on;

heatmapHandle = imagesc(gradCAMFull);
set(heatmapHandle,'AlphaData',0.45 * gradCAMFull);

axis image off;
colormap jet;
colorbar;

title('Fundus + Grad-CAM Overlay','FontWeight','bold');
hold off;

%% Overall title
if isfield(result,'predictedClass')
    predictedClass = string(result.predictedClass);
else
    predictedClass = "Unknown";
end

if isfield(result,'ICDRGrade')
    grade = result.ICDRGrade;
else
    grade = NaN;
end

if isfield(result,'referableDR')
    if result.referableDR
        referralText = "REFERABLE DR";
    else
        referralText = "NON-REFERABLE DR";
    end
else
    referralText = "";
end

sgtitle(sprintf('DR: %s | Grade %g | %s', ...
    predictedClass,grade,referralText), ...
    'FontWeight','bold');

end
