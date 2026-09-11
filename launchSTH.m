%% launchSTH - one-click SIH DR Screening launcher (MATLAB Desktop / Online)
% Usage: open in MATLAB and press Run, or type launchSTH in Command Window.
projectRoot = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(projectRoot)));
fprintf('Paths added for: %s\n', projectRoot);
runDRDemoV2;
