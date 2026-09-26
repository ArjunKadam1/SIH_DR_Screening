function varargout = probeLoss_TrackB_20260918(varargin)
% Probe: report what trainnet passes to a custom loss function.
fprintf('PROBE nargin=%d | ', nargin);
for i = 1:nargin
    v = varargin{i};
    if isdlarray(v)
        fprintf('arg%d=dlarray(%s) ', i, mat2str(size(extractdata(v))));
    elseif isnumeric(v)
        fprintf('arg%d=%s(%s) ', i, class(v), mat2str(size(v)));
    else
        fprintf('arg%d=%s ', i, class(v));
    end
end
fprintf('\n');
error('PROBE_DONE_STOP');
end
