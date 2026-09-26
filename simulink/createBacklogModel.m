%% createBacklogModel - camp-day backlog simulator (Track B)
% dBacklog/dt = inflow - serviceRate, floored at zero (Saturation).
% inflow  = arrivalPerHour/3600 patients/sec (default 20/hr)
% service = 1/105 patients/sec ~= 34.3/hr
%           (5 min per referable x 35%% referable, averaged over all arrivals)
% AI 2.5s transport delay kept on the inflow branch (negligible at day
% scale, preserves the original arrival -> AI -> review story).
%
% HONEST TOY: averaged continuous flow, NOT discrete patients; single
% doctor; flat arrival rate. Base Simulink only (no SimEvents).
% Usage: createBacklogModel        % 20/hr day
%        createBacklogModel(40)    % stress day
function createBacklogModel(arrivalPerHour)
if nargin < 1 || isempty(arrivalPerHour)
    arrivalPerHour = 20;
end
model = 'DR_Backlog_Model';
if bdIsLoaded(model), close_system(model,0); end
if exist([model '.slx'],'file'), delete([model '.slx']); end
new_system(model); open_system(model);
add_block('simulink/Sources/Constant',[model '/ArrivalPerHour'], ...
    'Value',num2str(arrivalPerHour));
add_block('simulink/Math Operations/Gain',[model '/PerSecond'], ...
    'Gain','1/3600');
add_block('simulink/Continuous/Transport Delay',[model '/AI_Processing'], ...
    'DelayTime','2.5');
add_block('simulink/Sources/Constant',[model '/ServiceRate'], ...
    'Value','1/105');
add_block('simulink/Math Operations/Sum',[model '/NetFlow'], ...
    'Inputs','+-');
add_block('simulink/Continuous/Integrator',[model '/BacklogInt'], ...
    'InitialCondition','0');
add_block('simulink/Discontinuities/Saturation',[model '/FloorZero'], ...
    'UpperLimit','inf','LowerLimit','0');
add_block('simulink/Sinks/Scope',[model '/BacklogScope']);
add_block('simulink/Sinks/Display',[model '/BacklogDisplay']);
add_block('simulink/Sinks/To Workspace',[model '/LogBacklog'], ...
    'VariableName','backlogLog','SaveFormat','Array');
add_line(model,'ArrivalPerHour/1','PerSecond/1');
add_line(model,'PerSecond/1','AI_Processing/1');
add_line(model,'AI_Processing/1','NetFlow/1');
add_line(model,'ServiceRate/1','NetFlow/2');
add_line(model,'NetFlow/1','BacklogInt/1');
add_line(model,'BacklogInt/1','FloorZero/1');
add_line(model,'FloorZero/1','BacklogScope/1');
add_line(model,'FloorZero/1','BacklogDisplay/1');
add_line(model,'FloorZero/1','LogBacklog/1');
set_param(model,'StopTime','28800'); % 8-hour camp day in seconds
save_system(model);
fprintf(['Saved %s.slx | arrival %g/hr, service %.1f/hr, AI 2.5s/img.\n' ...
    'Assumptions: averaged flow, single doctor, flat arrivals.\n'], ...
    model, arrivalPerHour, 3600/105);
end
