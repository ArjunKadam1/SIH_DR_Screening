%% createThroughputModel - builds SIH throughput/bottleneck model
% Arrival -> AI inference -> Ophthalmologist review -> Queue/backlog
% All assumptions labeled; replace with measured inference time.
function createThroughputModel()
model = 'DR_Throughput_Model';
if bdIsLoaded(model), close_system(model,0); end
if exist([model '.slx'],'file'), delete([model '.slx']); end
new_system(model); open_system(model);
% Parameters (ASSUMPTIONS - label in pitch)
arrivalPerHour = 20; aiSecPerImage = 2.5; reviewMinPerReferable = 5; referableRate = 0.35;
% Use SimEvents-free discrete model: Constant arrivals, delays, scope
add_block('simulink/Sources/Constant',[model '/ArrivalRate'], 'Value', num2str(arrivalPerHour));
add_block('simulink/Continuous/Transport Delay',[model '/AI_Processing'], 'TimeDelay', num2str(aiSecPerImage));
add_block('simulink/Continuous/Transport Delay',[model '/Review_Queue'], 'TimeDelay', num2str(reviewMinPerReferable*60*referableRate));
add_block('simulink/Sinks/Scope',[model '/BacklogScope']);
add_block('simulink/Sinks/Display',[model '/ThroughputDisplay']);
add_line(model,'ArrivalRate/1','AI_Processing/1');
add_line(model,'AI_Processing/1','Review_Queue/1');
add_line(model,'Review_Queue/1','BacklogScope/1');
add_line(model,'Review_Queue/1','ThroughputDisplay/1');
set_param(model,'StopTime','8'); % 8-hour camp day
save_system(model); 
fprintf('Saved %s.slx\nAssumptions: arrival %g/hr, AI %.1fs/img, review %.0fmin x %.0f%% referable.\n', ...
    model, arrivalPerHour, aiSecPerImage, reviewMinPerReferable, referableRate*100);
fprintf('Measure real inference with: tic; classify(net,img); toc  then update TimeDelay.\n');
end
