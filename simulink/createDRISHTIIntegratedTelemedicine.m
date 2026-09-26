function createDRISHTIIntegratedTelemedicine

model = 'DRISHTI_Integrated_Telemedicine_System';

%% Clean previous version
if bdIsLoaded(model)
    close_system(model,0);
end

if isfile(fullfile(pwd,'simulink',[model '.slx']))
    delete(fullfile(pwd,'simulink',[model '.slx']));
end

new_system(model);
open_system(model);

set_param(model,'Solver','FixedStepDiscrete');
set_param(model,'FixedStep','1');
set_param(model,'StopTime','1');
set_param(model,'SaveOutput','on');
set_param(model,'OutputSaveName','yout');
set_param(model,'ReturnWorkspaceOutputs','on');

%% ============================================================
% PARAMETERS
% =============================================================

patientsPerYear = 100000;
operatingDays = 300;
hoursPerDay = 8;

patientsPerHour = patientsPerYear/(operatingDays*hoursPerDay);

imageSizeMB = 2;
bandwidthMbps = 10;

aiUnits = 1;
aiProcessingSeconds = 2.4;

referableRate = 0.20;

reviewers = 2;
reviewSeconds = 60;

% Representative screening case
qualityScore = 0.85;
gradeInput = 2;
modelScore = 0.5395;

assignin('base','DRISHTI_patientsPerYear',patientsPerYear);
assignin('base','DRISHTI_operatingDays',operatingDays);
assignin('base','DRISHTI_hoursPerDay',hoursPerDay);
assignin('base','DRISHTI_patientsPerHour',patientsPerHour);
assignin('base','DRISHTI_imageSizeMB',imageSizeMB);
assignin('base','DRISHTI_bandwidthMbps',bandwidthMbps);
assignin('base','DRISHTI_aiUnits',aiUnits);
assignin('base','DRISHTI_aiProcessingSeconds',aiProcessingSeconds);
assignin('base','DRISHTI_referableRate',referableRate);
assignin('base','DRISHTI_reviewers',reviewers);
assignin('base','DRISHTI_reviewSeconds',reviewSeconds);
assignin('base','DRISHTI_qualityScore',qualityScore);
assignin('base','DRISHTI_gradeInput',gradeInput);
assignin('base','DRISHTI_modelScore',modelScore);

%% ============================================================
% INPUT CONSTANT BLOCKS
% =============================================================

inputNames = {
    'Patients_Per_Hour'
    'Image_Size_MB'
    'Bandwidth_Mbps'
    'AI_Units'
    'AI_Time_sec'
    'Referable_Rate'
    'Reviewers'
    'Review_Time_sec'
    'Quality_Score'
    'Grade_Input'
    'Model_Score'
    };

inputValues = {
    'DRISHTI_patientsPerHour'
    'DRISHTI_imageSizeMB'
    'DRISHTI_bandwidthMbps'
    'DRISHTI_aiUnits'
    'DRISHTI_aiProcessingSeconds'
    'DRISHTI_referableRate'
    'DRISHTI_reviewers'
    'DRISHTI_reviewSeconds'
    'DRISHTI_qualityScore'
    'DRISHTI_gradeInput'
    'DRISHTI_modelScore'
    };

y = 40;

for k = 1:numel(inputNames)

    add_block( ...
        'simulink/Sources/Constant', ...
        [model '/' inputNames{k}], ...
        'Value',inputValues{k}, ...
        'Position',[30 y 190 y+30]);

    y = y + 48;

end

%% ============================================================
% 1. NETWORK / ACQUISITION FUNCTION
% =============================================================

networkBlock = [model '/Network_Acquisition'];

add_block( ...
    'simulink/User-Defined Functions/MATLAB Function', ...
    networkBlock, ...
    'Position',[300 100 570 270]);

rt = sfroot;
chart = rt.find('-isa','Stateflow.EMChart','Path',networkBlock);
chart = chart(1);

chart.Script = sprintf([ ...
'function [networkCapacity,load,ready] = networkModel(patients,imageMB,bwMbps)\n' ...
'networkCapacity = (bwMbps*3600)/(8*imageMB);\n' ...
'load = patients;\n' ...
'ready = double(load <= networkCapacity);\n' ...
'end\n']);

%% ============================================================
% 2. DRISHTI SCREENING FUNCTION
% =============================================================

screenBlock = [model '/DRISHTI_Screening'];

add_block( ...
    'simulink/User-Defined Functions/MATLAB Function', ...
    screenBlock, ...
    'Position',[650 100 940 310]);

rt = sfroot;
chart = rt.find('-isa','Stateflow.EMChart','Path',screenBlock);
chart = chart(1);

chart.Script = sprintf([ ...
'function [grade,score,referable,screeningTime] = screeningModel(networkReady,quality,gradeInput,modelScore)\n' ...
'grade = -1;\n' ...
'score = 0;\n' ...
'referable = 0;\n' ...
'screeningTime = 0;\n' ...
'\n' ...
'if networkReady == 0\n' ...
'    return;\n' ...
'end\n' ...
'\n' ...
'if quality < 0.40\n' ...
'    return;\n' ...
'end\n' ...
'\n' ...
'grade = gradeInput;\n' ...
'score = modelScore;\n' ...
'screeningTime = 2.4;\n' ...
'\n' ...
'if grade >= 2\n' ...
'    referable = 1;\n' ...
'end\n' ...
'end\n']);

%% ============================================================
% 3. RESOURCE / REVIEW FUNCTION
% =============================================================

resourceBlock = [model '/Resource_Analysis'];

add_block( ...
    'simulink/User-Defined Functions/MATLAB Function', ...
    resourceBlock, ...
    'Position',[1010 100 1320 340]);

rt = sfroot;
chart = rt.find('-isa','Stateflow.EMChart','Path',resourceBlock);
chart = chart(1);

chart.Script = sprintf([ ...
'function [aiCapacity,reviewCapacity,aiLoad,reviewLoad,bottleneck] = resourceModel(patients,networkCapacity,aiUnits,aiSec,refRate,reviewers,reviewSec)\n' ...
'aiCapacity = aiUnits*(3600/aiSec);\n' ...
'reviewCapacity = reviewers*(3600/reviewSec);\n' ...
'\n' ...
'aiLoad = min(patients,networkCapacity);\n' ...
'reviewLoad = aiLoad*refRate;\n' ...
'\n' ...
'bottleneck = 0;\n' ...
'\n' ...
'if patients > networkCapacity\n' ...
'    bottleneck = 1;\n' ...
'elseif aiLoad > aiCapacity\n' ...
'    bottleneck = 2;\n' ...
'elseif reviewLoad > reviewCapacity\n' ...
'    bottleneck = 3;\n' ...
'end\n' ...
'end\n']);

%% ============================================================
% Update model so function ports exist
% =============================================================

set_param(model,'SimulationCommand','update');

%% ============================================================
% CONNECT NETWORK INPUTS
% =============================================================

add_line(model,'Patients_Per_Hour/1','Network_Acquisition/1',...
    'autorouting','on');

add_line(model,'Image_Size_MB/1','Network_Acquisition/2',...
    'autorouting','on');

add_line(model,'Bandwidth_Mbps/1','Network_Acquisition/3',...
    'autorouting','on');

%% ============================================================
% NETWORK → SCREENING
% =============================================================

add_line(model,'Network_Acquisition/3','DRISHTI_Screening/1',...
    'autorouting','on');

add_line(model,'Quality_Score/1','DRISHTI_Screening/2',...
    'autorouting','on');

add_line(model,'Grade_Input/1','DRISHTI_Screening/3',...
    'autorouting','on');

add_line(model,'Model_Score/1','DRISHTI_Screening/4',...
    'autorouting','on');

%% ============================================================
% RESOURCE INPUTS
% =============================================================

add_line(model,'Patients_Per_Hour/1','Resource_Analysis/1',...
    'autorouting','on');

add_line(model,'Network_Acquisition/1','Resource_Analysis/2',...
    'autorouting','on');

add_line(model,'AI_Units/1','Resource_Analysis/3',...
    'autorouting','on');

add_line(model,'AI_Time_sec/1','Resource_Analysis/4',...
    'autorouting','on');

add_line(model,'Referable_Rate/1','Resource_Analysis/5',...
    'autorouting','on');

add_line(model,'Reviewers/1','Resource_Analysis/6',...
    'autorouting','on');

add_line(model,'Review_Time_sec/1','Resource_Analysis/7',...
    'autorouting','on');

%% ============================================================
% OUTPUTS
% =============================================================

outputNames = {
    'Network_Capacity'
    'Network_Load'
    'Network_Ready'
    'DR_Grade'
    'Model_Score'
    'Referable_DR'
    'Screening_Time'
    'AI_Capacity'
    'Review_Capacity'
    'AI_Load'
    'Review_Load'
    'Bottleneck'
    };

% Network outputs
outputSources = {
    'Network_Acquisition/1'
    'Network_Acquisition/2'
    'Network_Acquisition/3'
    'DRISHTI_Screening/1'
    'DRISHTI_Screening/2'
    'DRISHTI_Screening/3'
    'DRISHTI_Screening/4'
    'Resource_Analysis/1'
    'Resource_Analysis/2'
    'Resource_Analysis/3'
    'Resource_Analysis/4'
    'Resource_Analysis/5'
    };

x = 1450;
y = 40;

for k = 1:numel(outputNames)

    block = [model '/' outputNames{k} '_Out'];

    add_block( ...
        'simulink/Sinks/Out1', ...
        block, ...
        'Position',[x y x+80 y+25]);

    add_line(model, ...
        outputSources{k}, ...
        [outputNames{k} '_Out/1'], ...
        'autorouting','on');

    y = y + 45;

end

%% ============================================================
% SAVE
% =============================================================

save_system(model,...
    fullfile(pwd,'simulink',[model '.slx']));

open_system(model);

fprintf('\n============================================\n');
fprintf('DRISHTI INTEGRATED TELEMEDICINE MODEL\n');
fprintf('============================================\n');
fprintf('Annual workload: %.0f patients\n',patientsPerYear);
fprintf('Workload: %.2f patients/hour\n',patientsPerHour);
fprintf('Network: %.2f Mbps\n',bandwidthMbps);
fprintf('AI units: %.0f\n',aiUnits);
fprintf('Reviewers: %.0f\n',reviewers);
fprintf('============================================\n');

end