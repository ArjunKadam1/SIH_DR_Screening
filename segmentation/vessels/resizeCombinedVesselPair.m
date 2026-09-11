function dataOut = resizeCombinedVesselPair(dataIn)

    I = dataIn{1};
    C = dataIn{2};

    disp("Inside transform:")
    disp(categories(C))
    disp(sum(C(:) == 'vessel'))

    [I, C] = resizeVesselPair(I, C);

    dataOut = {I, C};

end