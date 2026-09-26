function createDRISHTIResourceScenarios()

model = 'DRISHTI_Resource_Allocation_Scenarios';

%% Clean start
if bdIsLoaded(model)
    close_system(model,0);
end

if isfile([model '.slx'])
    delete([model '.slx']);
end

%% Create model
new_system(model);
open_system(model);

set_param(model, ...
    'Solver','FixedStepDiscrete', ...
    'FixedStep','1', ...
    'StopTime','1');

%% Scenario parameters
% {Name, Bandwidth Mbps, AI Units, AI Time sec, Reviewers, Review Time sec}

scenarios = {
    'Baseline',       10.0, 1,   2.4,   2,  60.0
    'Low_Bandwidth',  0.1,  1,   2.4,   2,  60.0
    'Limited_AI',     10.0, 1, 120.0,   2,  60.0
    'Limited_Review', 10.0, 1,   2.4,   1, 600.0
};

patients = 100000/(300*8);
imageMB  = 2.0;
refRate  = 0.20;

%% Create four scenario blocks
for s = 1:4

    name      = scenarios{s,1};
    bwMbps    = scenarios{s,2};
    aiUnits   = scenarios{s,3};
    aiSec     = scenarios{s,4};
    reviewers = scenarios{s,5};
    reviewSec = scenarios{s,6};

    x = 40 + (s-1)*300;

    blockPath = [model '/' name];

    %% Top-level subsystem
    add_block('simulink/Ports & Subsystems/Subsystem', ...
        blockPath, ...
        'Position',[x 150 x+220 350]);

    %% Find whatever blocks Simulink created
    existing = find_system(blockPath, ...
        'SearchDepth',1, ...
        'Type','Block');

    % Delete any default blocks individually
    for k = 1:numel(existing)
        if ~strcmp(existing{k},blockPath)
            delete_block(existing{k});
        end
    end

    %% Constants
    values = [
        patients
        imageMB
        bwMbps
        aiUnits
        aiSec
        refRate
        reviewers
        reviewSec
    ];

    names = {
        'Patients'
        'ImageMB'
        'Bandwidth'
        'AIUnits'
        'AISec'
        'RefRate'
        'Reviewers'
        'ReviewSec'
    };

    for k = 1:8

        add_block('simulink/Sources/Constant', ...
            [blockPath '/' names{k}], ...
            'Value',num2str(values(k)), ...
            'Position',[20 20+(k-1)*45 80 40+(k-1)*45]);

    end

    %% MATLAB Function
    fcnPath = [blockPath '/Capacity_Model'];

    add_block('simulink/User-Defined Functions/MATLAB Function', ...
        fcnPath, ...
        'Position',[180 120 390 380]);

    %% MATLAB Function code
    rt = sfroot;
    chart = rt.find('-isa','Stateflow.EMChart','Path',fcnPath);

    chart.Script = [
        "function [netCap,aiCap,reviewCap,netLoad,aiLoad,reviewLoad,bottleneck] = Capacity_Model(patients,imageMB,bwMbps,aiUnits,aiSec,refRate,reviewers,reviewSec)" newline ...
        "netCap = (bwMbps*3600)/(8*imageMB);" newline ...
        "aiCap = aiUnits*(3600/aiSec);" newline ...
        "reviewCap = reviewers*(3600/reviewSec);" newline ...
        "netLoad = patients;" newline ...
        "aiLoad = patients;" newline ...
        "reviewLoad = patients*refRate;" newline ...
        "if netLoad > netCap" newline ...
        "    bottleneck = 1;" newline ...
        "elseif aiLoad > aiCap" newline ...
        "    bottleneck = 2;" newline ...
        "elseif reviewLoad > reviewCap" newline ...
        "    bottleneck = 3;" newline ...
        "else" newline ...
        "    bottleneck = 0;" newline ...
        "end" newline ...
        "end"
    ];

    %% Connect 8 inputs
    for k = 1:8
        add_line(blockPath, ...
            [names{k} '/1'], ...
            ['Capacity_Model/' num2str(k)], ...
            'autorouting','on');
    end

    %% Outputs
    outputNames = {
        'Network_Capacity'
        'AI_Capacity'
        'Review_Capacity'
        'Network_Load'
        'AI_Load'
        'Review_Load'
        'Bottleneck'
    };

    for k = 1:7

        outPath = [blockPath '/' outputNames{k}];

        add_block('simulink/Sinks/Out1', ...
            outPath, ...
            'Position',[450 30+(k-1)*45 500 50+(k-1)*45]);

        add_line(blockPath, ...
            ['Capacity_Model/' num2str(k)], ...
            [outputNames{k} '/1'], ...
            'autorouting','on');

    end

end

%% Save
save_system(model);

open_system(model);

disp(' ');
disp('==========================================');
disp('DRISHTI RESOURCE SCENARIOS CREATED');
disp('==========================================');
disp('Baseline');
disp('Low_Bandwidth');
disp('Limited_AI');
disp('Limited_Review');
disp('==========================================');

end