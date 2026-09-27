%% exportResNetToONNX - MATLAB -> ONNX for Python deployment
% Exports the LIVE grader: handheld R50 aptosidrid4 (4-class merged
% No_DR/Mild/Moderate/SevereProlif, 224x224x3 input, var trainedNetHandheld).
% Run in MATLAB R2026a with Deep Learning Toolbox. Traced metrics in MODELS.md.
% NOTE: the Python side must feed *enhanced* 224 inputs (mirroring
% screenFundusImage) and expect 4 classes, not the old V1 5-class layout.
S = load('C:/Users/SHIVANYA SALES/Desktop/DR tejas/HandheldDR/results/handheld_resnet50_aptosidrid4.mat', ...
    'trainedNetHandheld');
exportONNXNetwork(S.trainedNetHandheld, fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    'results', 'handheld_r50_aptosidrid4.onnx'));
fprintf('Exported handheld_r50_aptosidrid4.onnx - copy to deploy/python_app/model/handheld_r50_aptosidrid4.onnx\n');
