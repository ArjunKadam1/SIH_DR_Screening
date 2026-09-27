%% exportResNetToONNX - MATLAB -> ONNX for Python deployment
% Exports the LIVE grader: handheld R50 aptosidrid4 (4-class merged
% No_DR/Mild/Moderate/SevereProlif, 224x224x3 input, var trainedNetHandheld).
% Run in MATLAB R2026a with Deep Learning Toolbox + ONNX Converter add-on.
% Traced metrics in MODELS.md. The R50 file is located portably: an ignored
% local models/ folder first, else an interactive file picker (no absolute
% paths in this script).
% NOTE: the Python side must feed *enhanced* 224 inputs (mirroring
% screenFundusImage) and expect 4 classes, not the old V1 5-class layout.
projRoot = fileparts(fileparts(mfilename('fullpath')));
cand = fullfile(projRoot, 'models', 'handheld_resnet50_aptosidrid4.mat');
if ~isfile(cand)
    [f, p] = uigetfile('*.mat', ...
        'Select handheld R50 .mat (must contain trainedNetHandheld)');
    if isequal(f, 0)
        error('Export cancelled: no R50 file selected.');
    end
    cand = fullfile(p, f);
end
S = load(cand, 'trainedNetHandheld');
exportONNXNetwork(S.trainedNetHandheld, fullfile(projRoot, ...
    'results', 'handheld_r50_aptosidrid4.onnx'));
fprintf('Exported handheld_r50_aptosidrid4.onnx - copy to deploy/python_app/model/handheld_r50_aptosidrid4.onnx\n');
