%% exportResNetToONNX - MATLAB -> ONNX for Python deployment
% Run in MATLAB R2026a with Deep Learning Toolbox.
S = load(fullfile(fileparts(fileparts(mfilename('fullpath'))),'results','trained_resnet18.mat'),'trainedNet');
exportONNXNetwork(S.trainedNet, fullfile(fileparts(fileparts(mfilename('fullpath'))),'results','resnet18_dr.onnx'));
fprintf('Exported resnet18_dr.onnx - copy to deploy/python_app/model/resnet18_dr.onnx\n');
