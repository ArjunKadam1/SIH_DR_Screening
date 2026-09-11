function [savedFile, usedMethod] = generateScreeningPDF(pdfPath, imgName, qFlag, cVal, qStatus, result)
% One-page screening report. result = screenFundusImage struct.
% Reports all available fields from the V2 pipeline in a clean format.
% Produces a real PDF: mlreportgen if available, otherwise a figure-based
% PDF via built-in print. Falls back to .txt only if both fail.
% Outputs: savedFile = path actually written; usedMethod = 'pdf'|'txt'.

lines = buildReportLines(imgName, qFlag, cVal, qStatus, result);

% Attempt 1: mlreportgen Report Generator
import mlreportgen.report.* %#ok<NCOMMA>
try
    rpt = Report(pdfPath,'pdf');
    add(rpt, TitlePage('Title','DR Screening Report', ...
        'Subtitle','SIH 2026 Prototype - AI-assisted, clinical confirmation required', ...
        'Author','SIH_DR_Screening V2'));

    add(rpt, Heading1('Screening Summary'));

    % --- Image & Quality section ---
    add(rpt, Heading2('Image & Quality'));
    sz = result.originalImageSize;
    szStr = sprintf('%d x %d x %d', sz(1), sz(2), sz(3));
    if isfield(result,'qualityReason')
        qReason = char(string(result.qualityReason));
    else
        qReason = 'N/A';
    end
    t1 = Table({ ...
        'Image', imgName; ...
        'Image Size', szStr; ...
        'Quality (V1)', sprintf('%s (%.2f)', qFlag, cVal); ...
        'Quality (V2)', char(qStatus); ...
        'Quality Reason', qReason; ...
        'Pipeline', char(string(result.pipelineVersion))});
    t1.Style = {Border('solid')}; add(rpt, t1);

    % --- Classification section ---
    add(rpt, Heading2('Classification'));
    if isfield(result,'calScorePercent')
        confLine = sprintf('%.2f%%', result.modelScorePercent);
        calLine  = sprintf('%.2f%% (temp=%.2f)', result.calScorePercent, result.calTemperature);
        marginLine = sprintf('%.1f pp', result.scoreMargin);
        entropyLine = sprintf('%.3f', result.scoreEntropy);
        reliabilityLine = char(string(result.reliabilityFlag));
    else
        confLine = sprintf('%.2f%%', result.modelScorePercent);
        calLine  = 'N/A';
        marginLine = 'N/A';
        entropyLine = 'N/A';
        reliabilityLine = 'N/A';
    end
    if isfield(result,'referableDR')
        referableLine = char(string(result.referableDR));
    else
        referableLine = 'N/A';
    end
    t2 = Table({ ...
        'Predicted Class', char(string(result.predictedClass)); ...
        'Severity', char(string(result.severity)); ...
        'ICDR Grade', num2str(result.ICDRGrade); ...
        'Confidence (raw)', confLine; ...
        'Confidence (calibrated)', calLine; ...
        'Score Margin', marginLine; ...
        'Score Entropy', entropyLine; ...
        'Reliability', reliabilityLine; ...
        'Referable DR', referableLine});
    t2.Style = {Border('solid')}; add(rpt, t2);

    % --- Screening Decision section ---
    add(rpt, Heading2('Screening Decision'));
    if isfield(result,'ttaUsed') && result.ttaUsed
        ttaLine = sprintf('Yes (%d views)', result.ttaViews);
    else
        ttaLine = 'No';
    end
    if isfield(result,'secondOpinionConsulted') && result.secondOpinionConsulted
        soLine = 'Yes';
        soReferLine = char(string(result.secondOpinionRefer));
        escalatedLine = char(string(result.escalated));
    else
        soLine = 'No';
        soReferLine = 'N/A';
        escalatedLine = 'No';
    end
    t3 = Table({ ...
        'Recommendation', char(string(result.referralRecommendation)); ...
        'TTA Used', ttaLine; ...
        'Second Opinion Consulted', soLine; ...
        'Second Opinion Refer', soReferLine; ...
        'Escalated', escalatedLine});
    t3.Style = {Border('solid')}; add(rpt, t3);

    % --- Explainability section ---
    add(rpt, Heading2('Explainability'));
    if isfield(result,'gradCAMAvailable') && result.gradCAMAvailable
        gradcamLine = 'Generated';
    else
        gradcamLine = 'Not available';
    end
    t4 = Table({ ...
        'Grad-CAM', gradcamLine; ...
        'Feature Layer', char(string(result.gradCAMFeatureLayer)); ...
        'Reduction Layer', char(string(result.gradCAMReductionLayer)); ...
        'Note', char(string(result.gradCAMNote))});
    t4.Style = {Border('solid')}; add(rpt, t4);

    % --- Disclaimer ---
    add(rpt, Paragraph(''));
    add(rpt, Paragraph('Model V1: test acc 78.83%, referable sens 84.75% spec 97.54%. Prototype public-test, NOT clinical validation. Overfit noted: train 96.88% vs val 76.55%.'));
    add(rpt, Paragraph('AI-assisted screening prototype. Clinical confirmation is recommended before any clinical decision.'));

    close(rpt);
    savedFile = pdfPath; usedMethod = 'pdf';
    return;
catch ME1
    % Attempt 2: figure-based PDF via built-in print (no toolbox needed)
    try
        makeFigurePdf(pdfPath, lines);
        savedFile = pdfPath; usedMethod = 'pdf';
        return;
    catch ME2
        % Attempt 3: .txt fallback
        txtPath = strrep(pdfPath,'.pdf','.txt');
        fid = fopen(txtPath,'w');
        if fid > 0
            for k = 1:numel(lines)
                fprintf(fid,'%s\n', lines{k});
            end
            fclose(fid);
        end
        savedFile = txtPath; usedMethod = 'txt';
        warning('Report Generator + figure PDF failed - wrote .txt fallback.\n  mlreportgen: %s\n  print: %s', ME1.message, ME2.message);
    end
end
end

function makeFigurePdf(pdfPath, lines)
% MAKEFIGUREPDF - renders the report lines onto a white figure and prints
% to a real PDF using built-in print. Works without any extra toolbox.
fig = figure('Visible','off','Color','w', ...
    'PaperUnits','points','PaperPosition',[0 0 612 792], ...
    'Units','normalized');
ax = axes(fig,'Visible','off','Position',[0.02 0.02 0.96 0.96]);
xlim(ax,[0 1]); ylim(ax,[0 1]);
ax.YDir = 'normal';
n = numel(lines);
lineH = 0.985 - (0:(n-1)) * (0.92/n); % top-down, small gaps folded in
for k = 1:n
    lk = lines{k};
    isSep   = ~isempty(lk) && all(lk == '=');
    isDash  = ~isempty(lk) && all(lk == '-');
    isTitle = contains(lk,'DR SCREENING REPORT');
    if isSep || isDash
        fontW = 'normal'; fs = 6; colr = [0.6 0.6 0.6];
    elseif isTitle
        fontW = 'bold';   fs = 14; colr = [0 0 0];
    elseif ~isempty(strfind(lk,':')) %#ok<STREMP>
        fontW = 'normal'; fs = 11; colr = [0 0 0];
    else
        fontW = 'normal'; fs = 11; colr = [0.2 0.2 0.2];
    end
    text(ax, 0.03, lineH(k), lk, 'FontName','Courier', ...
        'FontSize',fs, 'FontWeight',fontW, 'Color',colr, ...
        'Interpreter','none', 'VerticalAlignment','top');
end
print(fig, pdfPath, '-dpdf', '-r150');
close(fig);
end

function lines = buildReportLines(imgName, qFlag, cVal, qStatus, result)
% BUILDREPORTLINES - returns the full report as a cell array of lines,
% shared by the figure-PDF renderer and the .txt fallback.
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
if isfield(result,'pipelineVersion')
    L{end+1} = sprintf('Pipeline      : %s', char(string(result.pipelineVersion)));
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
if isfield(result,'ttaUsed')
    if result.ttaUsed
        L{end+1} = sprintf('TTA Used      : Yes (%d views)', result.ttaViews);
    else
        L{end+1} = 'TTA Used      : No';
    end
end
if isfield(result,'secondOpinionConsulted')
    if result.secondOpinionConsulted
        L{end+1} = sprintf('Second Opinion: Consulted (refer=%s, escalated=%s)', ...
            char(string(result.secondOpinionRefer)), char(string(result.escalated)));
    else
        L{end+1} = 'Second Opinion: Not consulted';
    end
end
L{end+1} = dash;

% Explainability
if isfield(result,'gradCAMAvailable') && result.gradCAMAvailable
    L{end+1} = 'Grad-CAM      : Generated';
else
    L{end+1} = 'Grad-CAM      : Not available';
end
if isfield(result,'gradCAMFeatureLayer')
    L{end+1} = sprintf('Feature Layer : %s', char(string(result.gradCAMFeatureLayer)));
end
if isfield(result,'gradCAMReductionLayer')
    L{end+1} = sprintf('Reduction     : %s', char(string(result.gradCAMReductionLayer)));
end
if isfield(result,'gradCAMNote')
    L{end+1} = sprintf('Grad-CAM Note : %s', char(string(result.gradCAMNote)));
end
L{end+1} = sep;
L{end+1} = '';

% Disclaimer
L{end+1} = 'Note: AI-assisted screening prototype.';
L{end+1} = 'Clinical confirmation is recommended before any clinical decision.';
L{end+1} = 'Model V1: test acc 78.83%, referable sens 84.75% spec 97.54%.';
L{end+1} = 'Prototype public-test, NOT clinical validation.';
L{end+1} = sep;

lines = L(:);
end