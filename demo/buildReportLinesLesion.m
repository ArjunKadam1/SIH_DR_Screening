function lines = buildReportLinesLesion(imgName, qFlag, cVal, qStatus, result)
% BUILDREPORTLINESLESION - shared report-lines builder (pure function).
%
% Single source of truth for the screening report text, used by BOTH
% generateScreeningPDF_Lesion (PDF export) and RetinaAIApp_Lesion (in-app
% Report tab preview), so preview and PDF can never drift apart.
% Extracted verbatim from generateScreeningPDF_Lesion; behavior identical.
% Accuracy 82.30 traced: HandheldDR/reports/all_models_report.txt
% (APTOS-test n=548, 13-Sep-2026). Prototype-labeled throughout.
sep = '========================================';
dash = '----------------------------------------';
L = {};
L{end+1} = sep;
L{end+1} = '          DR SCREENING REPORT';
L{end+1} = sep;
L{end+1} = '';

% Image & Quality
L{end+1} = sprintf('Image         : %s', imgName);
if isfield(result,'originalImageSize')
    sz = result.originalImageSize;
    L{end+1} = sprintf('Image Size    : %d x %d x %d', sz(1), sz(2), sz(3));
end
L{end+1} = sprintf('Quality (V1)  : %s (%.2f)', qFlag, cVal);
L{end+1} = sprintf('Quality (V2)  : %s', qStatus);
if isfield(result,'qualityReason')
    L{end+1} = sprintf('Quality Reason: %s', char(string(result.qualityReason)));
end
L{end+1} = dash;

% Classification
L{end+1} = sprintf('Predicted DR  : %s', char(string(result.predictedClass)));
if isfield(result,'severity')
    L{end+1} = sprintf('Severity      : %s', char(string(result.severity)));
end
L{end+1} = sprintf('ICDR Grade    : %g', result.ICDRGrade);
L{end+1} = sprintf('Confidence    : %.2f%%', result.modelScorePercent);
if isfield(result,'calScorePercent')
    L{end+1} = sprintf('Calibrated    : %.2f%% (temp=%.2f)', result.calScorePercent, result.calTemperature);
    L{end+1} = sprintf('Margin        : %.1f pp', result.scoreMargin);
    L{end+1} = sprintf('Entropy       : %.3f', result.scoreEntropy);
    L{end+1} = sprintf('Reliability   : %s', char(string(result.reliabilityFlag)));
end
if isfield(result,'referableDR')
    L{end+1} = sprintf('Referable DR  : %s', char(string(result.referableDR)));
end
L{end+1} = dash;

% Screening Decision
L{end+1} = sprintf('Screening     : %s', char(string(result.referralRecommendation)));
L{end+1} = dash;

% Explainability
if isfield(result,'gradCAMAvailable') && result.gradCAMAvailable
    L{end+1} = 'Grad-CAM      : Generated';
else
    L{end+1} = 'Grad-CAM      : Not available';
end
if isfield(result,'gradCAMReductionLayer')
    L{end+1} = sprintf('Reduction     : %s', char(string(result.gradCAMReductionLayer)));
end
L{end+1} = dash;

% Lesion Evidence
L{end+1} = 'Lesion Evidence:';
if isfield(result,'maAvailable') && result.maAvailable
    L{end+1} = sprintf('MA dots       : %d @ thr %.2f (%.1fs)', ...
        result.maCount, result.maThr, result.maTimeSec);
elseif isfield(result,'maAvailable')
    L{end+1} = 'MA            : not available (net missing or failed)';
else
    L{end+1} = 'MA            : not run (opt-in off)';
end
if isfield(result,'vesselAvailable') && result.vesselAvailable
    L{end+1} = sprintf('Vessel %%FOV    : %.2f%% @ thr %.2f (%.1fs)', ...
        result.vesselFracFOV, result.vesselThr, result.vesselTimeSec);
elseif isfield(result,'vesselAvailable')
    L{end+1} = 'Vessel        : not available — upload FOV mask or check net';
else
    L{end+1} = 'Vessel        : not run (opt-in off)';
end
if isfield(result,'lesionNote')
    L{end+1} = sprintf('Lesion note   : %s', char(string(result.lesionNote)));
end
L{end+1} = sep;
L{end+1} = '';

% Disclaimer
L{end+1} = 'Note: AI-assisted screening prototype.';
L{end+1} = 'Clinical confirmation is recommended before any clinical decision.';
L{end+1} = 'Model R50 aptosidrid4: APTOS-test acc 82.30%, referable sens 91.93% spec 95.08%.';
L{end+1} = 'MA ep2 pooled val Dice 0.42; P3-A vessel DRIVE-val dFix 0.37 / thinRecall ~0.73.';
L{end+1} = sep;

lines = L(:);
end
