function [savedFile, usedMethod] = generateScreeningPDF_Lesion(pdfPath, imgName, qFlag, cVal, qStatus, result)
% GENERATESCREENINGPDF_LESION - screening report + Lesion Evidence section.
%
% Clone of generateScreeningPDF for the new RetinaAIApp_Lesion front end.
% Old generateScreeningPDF.m is untouched (fallback). Behavior is identical
% when result has no lesion fields; when result carries maAvailable /
% vesselAvailable (added by runLesionEvidence via the new app), an extra
% "Lesion Evidence (prototype)" section is appended.
%
% Lesion labels: MA 0.42 val / P3-A vessel dFix 0.37 val, supporting
% evidence only, NOT clinical. Test-27 IDRiD untouched.

lines = buildReportLinesLesion(imgName, qFlag, cVal, qStatus, result);  % shared builder (also feeds in-app Report tab)

% Attempt 1: mlreportgen Report Generator
import mlreportgen.report.* %#ok<NCOMMA>
try
    rpt = Report(pdfPath,'pdf');
    add(rpt, TitlePage('Title','DR Screening Report', ...
        'Subtitle','SIH 2026 Prototype - AI-assisted, clinical confirmation required', ...
        'Author','SIH_DR_Screening V2 + Lesion'));

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
        'Quality Reason', qReason});
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
    t3 = Table({ ...
        'Recommendation', char(string(result.referralRecommendation))});
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
        'Reduction Layer', char(string(result.gradCAMReductionLayer))});
    t4.Style = {Border('solid')}; add(rpt, t4);

    % --- Lesion Evidence section (only when lesion fields exist) ---
    add(rpt, Heading2('Lesion Evidence'));
    lt = lesionTableRows(result);
    t5 = Table(lt);
    t5.Style = {Border('solid')}; add(rpt, t5);

    % --- Disclaimer ---
    add(rpt, Paragraph(''));
    add(rpt, Paragraph('Model R50 aptosidrid4: APTOS-test acc 82.30%, referable sens 91.93% spec 95.08%. MA ep2 pooled val Dice 0.42; P3-A vessel DRIVE-val dFix 0.37 / thinRecall ~0.73.'));  % 82.30 traced: HandheldDR/reports/all_models_report.txt (APTOS-test n=548, 13-Sep-2026)
    add(rpt, Paragraph('AI-assisted screening prototype. Clinical confirmation is recommended before any clinical decision.'));

    close(rpt);
    savedFile = pdfPath; usedMethod = 'pdf';
    return;
catch ME1
    % Attempt 2: figure-based PDF via built-in print (no toolbox needed)
    try
        makeFigurePdfLesion(pdfPath, lines);
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

function rows = lesionTableRows(result)
% LESIONTABLEROWS - 2-col cell for the mlreportgen lesion table.
rows = {'MA available', 'Not run (opt-in checkbox off or net missing)'};
try
    if isfield(result,'maAvailable') && result.maAvailable
        maN = result.maCount;
        maT = result.maThr;
        maTi = result.maTimeSec;
        rows(end+1,:) = {'MA dots (@thr)', sprintf('%d @ %.2f (%.1fs)', maN, maT, maTi)}; %#ok<AGROW>
    elseif isfield(result,'maAvailable')
        rows(end+1,:) = {'MA', 'Ran, none found or failed — see note'}; %#ok<AGROW>
    end
    if isfield(result,'vesselAvailable') && result.vesselAvailable
        rows(end+1,:) = {'Vessel %FOV (@thr)', ... %#ok<AGROW>
            sprintf('%.2f%% @ %.2f (%.1fs)', result.vesselFracFOV, result.vesselThr, result.vesselTimeSec)};
    elseif isfield(result,'vesselAvailable')
        rows(end+1,:) = {'Vessel', 'Not available — upload FOV mask or check net'}; %#ok<AGROW>
    end
    if isfield(result,'lesionNote')
        rows(end+1,:) = {'Lesion note', char(string(result.lesionNote))}; %#ok<AGROW>
    else
        rows(end+1,:) = {'Lesion note', 'MA 0.42 val / P3-A vessel dFix 0.37 val. Prototype, NOT clinical.'}; %#ok<AGROW>
    end
catch
    rows = {'Lesion Evidence', 'Unavailable (report guard)'};
end
end

function makeFigurePdfLesion(pdfPath, lines)
% MAKEFIGUREPDFLESION - renders the report lines onto a white figure.
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
