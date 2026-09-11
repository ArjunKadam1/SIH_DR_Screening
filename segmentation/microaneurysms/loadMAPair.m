function dataOut = loadMAPair(dataIn)
% loadMAPair
% Converts a patch-index entry into:
% {RGB image, categorical MA mask}

entry = dataIn{1};

[I,C] = readMAPatch(entry);

dataOut = {I,C};

end