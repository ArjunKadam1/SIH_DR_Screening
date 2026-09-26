function createDRISHTIDistrictSimulationV2

model = 'DRISHTI_District_Telemedicine_Simulation';

if bdIsLoaded(model)
    close_system(model,0);
end

new_system(model);
open_system(model);

%% Simulation settings
set_param(model,'Solver','FixedStepDiscrete');
set_param(model,'FixedStep','1');
set_param(model,'StopTime','1');

%% -------------------------------------------------
% DISTRICT PARAMETERS
% --------------------------------------------------

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

%% Save parameters to base workspace

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

%% -------------------------------------------------
% INPUT CONSTANTS
% --------------------------------------------------

inputNames = {
    'Patients_Per_Hour'
    'Image_Size_MB'
    'Bandwidth_Mbps'
    'AI_Units'
    'AI_Time_sec'
    'Referable_Rate'
    'Reviewers'
    'Review_Time_sec'
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
    };

y = 60;

for k = 1:numel(inputNames)

    blockPath = [model '/' inputNames{k}];

    add_block( ...
        'simulink/Sources/Constant', ...
        blockPath, ...
        'Value',inputValues{k}, ...
        'Position',[40 y 190 y+30]);

    y = y + 60;

end

%% -------------------------------------------------
% DISTRICT CAPACITY MATLAB FUNCTION
% --------------------------------------------------

functionPath = [model '/District_Capacity_Model'];

add_block( ...
    'simulink/User-Defined Functions/MATLAB Function', ...
    functionPath, ...
    'Position',[350 170 650 500]);

functionCode = [
"function [networkCapacity,aiCapacity,reviewCapacity,networkLoad,aiLoad,reviewLoad,bottleneck] = capacityModel(" newline ...
"patients,imageMB,bwMbps,aiUnits,aiSec,refRate,reviewers,reviewSec)" newline ...
"" newline ...
"% Network capacity: images/hour" newline ...
"networkCapacity = (bwMbps*3600)/(8*imageMB);" newline ...
"" newline ...
"% AI capacity: images/hour" newline ...
"aiCapacity = aiUnits*(3600/aiSec);" newline ...
"" newline ...
"% Human review capacity: cases/hour" newline ...
"reviewCapacity = reviewers*(3600/reviewSec);" newline ...
"" newline ...
"% Workload entering each stage" newline ...
"networkLoad = patients;" newline ...
"aiLoad = patients;" newline ...
"reviewLoad = patients*refRate;" newline ...
"" newline ...
"% Bottleneck code" newline ...
"% 0 = no bottleneck" newline ...
"% 1 = network" newline ...
"% 2 = AI processing" newline ...
"% 3 = human review" newline ...
"" newline ...
"if networkLoad > networkCapacity" newline ...
"    bottleneck = 1;" newline ...
"elseif aiLoad > aiCapacity" newline ...
"    bottleneck = 2;" newline ...
"elseif reviewLoad > reviewCapacity" newline ...
"    bottleneck = 3;" newline ...
"else" newline ...
"    bottleneck = 0;" newline ...
"end" newline ...
"end"
];

set_param(functionPath,'Script',functionCode);

%% -------------------------------------------------
% CONNECT INPUTS
% --------------------------------------------------

for k = 1:numel(inputNames)

    add_line( ...
        model, ...
        [inputNames{k} '/1'], ...
        ['District_Capacity_Model/' num2str(k)], ...
        'autorouting','on');

end

%% -------------------------------------------------
% OUTPUT PORTS
% --------------------------------------------------

outputNames = {
    'Network_Capacity'
    'AI_Capacity'
    'Review_Capacity'
    'Network_Load'
    'AI_Load'
    'Review_Load'
    'Bottleneck'
    };

y = 70;

for k = 1:numel(outputNames)

    blockPath = [model '/' outputNames{k} '_Out'];

    add_block( ...
        'simulink/Sinks/Out1', ...
        blockPath, ...
        'Position',[820 y 900 y+30]);

    add_line( ...
        model, ...
        ['District_Capacity_Model/' num2str(k)], ...
        [outputNames{k} '_Out/1'], ...
        'autorouting','on');

    y = y + 60;

end

%% -------------------------------------------------
% SAVE OUTPUTS
% --------------------------------------------------

set_param(model,'SaveOutput','on');
set_param(model,'OutputSaveName','yout');
set_param(model,'ReturnWorkspaceOutputs','on');

%% Save
save_system(model);

fprintf('\n');
fprintf('============================================\n');
fprintf('DRISHTI DISTRICT MODEL CREATED\n');
fprintf('============================================\n');
fprintf('Annual patients: %.0f\n',patientsPerYear);
fprintf('Patients/hour: %.2f\n',patientsPerHour);
fprintf('Image size: %.2f MB\n',imageSizeMB);
fprintf('Bandwidth: %.2f Mbps\n',bandwidthMbps);
fprintf('AI units: %.0f\n',aiUnits);
fprintf('Reviewers: %.0f\n',reviewers);
fprintf('============================================\n');

end