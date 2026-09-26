classdef RetinaAIApp < handle
% RETINAAIAPP - RETINA-AI judge-demo front end for the SIH DR pipeline.
%
% Explainable AI for Diabetic Retinopathy Screening (SIH 2026 prototype).
% Thin UI layer over the existing pipeline: upload -> screenFundusImage()
% -> quality card -> result card -> Grad-CAM views -> PDF report.
% No screening logic is duplicated here.
%
% Usage (MATLAB desktop or Online):
%   addpath('demo');  app = RetinaAIApp;
%
% Prototype wording only - nothing here is clinically validated.

properties (Access = private)
    Fig
    StatusLamp
    StatusLabel
    ImgAxes
    ExpAxes
    AnalyzeBtn
    ReportBtn
    ViewBtns            % 4x1 view buttons
    ResultSeverity
    ResultGrade
    ResultReferable
    ResultScore
    ResultReco
    ResultDetails
    QualityRows         % 3x2 labels: [valueLabel, flagLabel]
    QualityOverall
    QualityEnhance
    RejectBanner
    TrainedNet
    ProjectRoot
    OriginalImage = []
    ImgFileName = ""
    Result = struct()
    HasResult = false
    CVal = NaN
    QFlag = ""
end

methods (Access = public)
    function app = RetinaAIApp()
        % Derive project root from this file's location (works on any
        % machine / MATLAB Drive checkout, mirrors runDRDemoV2).
        app.ProjectRoot = fileparts(fileparts(mfilename('fullpath')));
        addpath(fullfile(app.ProjectRoot,'preprocessing'));
        addpath(fullfile(app.ProjectRoot,'quality'));
        addpath(fullfile(app.ProjectRoot,'demo'));
        addpath(fullfile(app.ProjectRoot,'classification'));

        app.buildUI();
        app.setStatus([0.2 0.7 0.2], 'Loading model…');
        drawnow;
        % Phase-2 swap (2026-09-25): handheld R50 aptosidrid4 live.
        % ROLLBACK: restore results/trained_resnet18.mat 'trainedNet' (kept).
        S = load('C:/Users/SHIVANYA SALES/Desktop/DR tejas/HandheldDR/results/handheld_resnet50_aptosidrid4.mat', ...
            'trainedNetHandheld');
        app.TrainedNet = S.trainedNetHandheld;
        app.setStatus([0.2 0.7 0.2], 'System Ready');
    end

    function out = headlessAnalyze(app, imgPath)
        % HEADLESSANALYZE - regression hook: loads image, runs the same
        % onAnalyze path as the Analyze button, returns visible card texts.
        app.OriginalImage = imread(imgPath);
        [~,f,e] = fileparts(imgPath);
        app.ImgFileName = string([f e]);
        app.AnalyzeBtn.Enable = 'on';
        app.onAnalyze();
        out = struct('severity',char(app.ResultSeverity.Text), ...
            'grade',char(app.ResultGrade.Text), ...
            'referable',char(app.ResultReferable.Text), ...
            'score',char(app.ResultScore.Text), ...
            'reco',char(app.ResultReco.Text), ...
            'quality',char(app.QualityOverall.Text), ...
            'enhance',char(app.QualityEnhance.Text), ...
            'banner',char(app.RejectBanner.Text), ...
            'qFocus',char(app.QualityRows(1,1).Text), ...
            'qIllum',char(app.QualityRows(2,1).Text), ...
            'qFov',char(app.QualityRows(3,1).Text), ...
            'details',char(app.ResultDetails.Text), ...
            'reportEnabled',char(app.ReportBtn.Enable));
    end
end

methods (Access = private)

    function buildUI(app)
        app.Fig = uifigure('Name','DRISHTI - DR Screening Assistant', ...
            'Position',[60 40 1360 820], 'Color',[0.92 0.94 0.96]);
        main = uigridlayout(app.Fig,[3 1]);
        main.RowHeight = {70,'1x',64};
        main.Padding = [8 8 8 8];

        % ---------- Header ----------
        navy = [0.07 0.18 0.32];
        hdr = uipanel(main,'BackgroundColor',navy,'BorderType','none');
        hg = uigridlayout(hdr,[2 2]);
        hg.ColumnWidth = {'1x',210};
        hg.RowHeight = {'1.2x','1x'};
        hg.BackgroundColor = navy;
        hg.Padding = [14 4 12 4];
        t = uilabel(hg,'Text','DRISHTI', ...
            'FontSize',23,'FontWeight','bold','FontColor','w');
        t.Layout.Row = 1; t.Layout.Column = 1;
        subt = uilabel(hg, ...
            'Text','Explainable AI for Diabetic Retinopathy Screening', ...
            'FontSize',12,'FontColor',[0.72 0.81 0.92]);
        subt.Layout.Row = 2; subt.Layout.Column = 1;
        sg = uigridlayout(hg,[1 2]);
        sg.ColumnWidth = {24,'1x'};
        sg.BackgroundColor = navy;
        sg.Layout.Row = [1 2]; sg.Layout.Column = 2;
        app.StatusLamp = uilamp(sg,'Color',[0.2 0.7 0.2]);
        app.StatusLabel = uilabel(sg,'Text','System Ready', ...
            'FontColor','w','FontSize',12,'FontWeight','bold');

        % ---------- Content: 3 columns ----------
        cg = uigridlayout(main,[1 3]);
        cg.ColumnWidth = {410,395,'1x'};
        cg.Padding = [0 4 0 4];

        % ----- Left: image -----
        left = uipanel(cg,'Title','FUNDUS IMAGE','FontWeight','bold');
        lg = uigridlayout(left,[4 1]);
        lg.RowHeight = {'1x',40,40,8};
        app.ImgAxes = uiaxes(lg);
        axis(app.ImgAxes,'off');
        title(app.ImgAxes,'No image loaded','Color',[0.5 0.5 0.5]);
        uibutton(lg,'Text','Upload Fundus Image', ...
            'ButtonPushedFcn',@(s,e)app.onUpload());
        app.AnalyzeBtn = uibutton(lg,'Text','ANALYZE IMAGE', ...
            'FontSize',13,'FontWeight','bold','Enable','off', ...
            'BackgroundColor',[0.05 0.38 0.62],'FontColor','w', ...
            'ButtonPushedFcn',@(s,e)app.onAnalyze());

        % ----- Middle: result + quality -----
        mid = uigridlayout(cg,[2 1]);
        mid.RowHeight = {'1.1x','1x'};
        res = uipanel(mid,'Title','AI SCREENING RESULT','FontWeight','bold');
        rg = uigridlayout(res,[7 2]);
        rg.ColumnWidth = {118,'1x'};
        rg.RowHeight = {30,30,30,30,66,4,48};
        app.addRow(rg,1,'DR Severity:', 'ResultSeverity', '—', 15, true);
        app.addRow(rg,2,'ICDR Grade:',  'ResultGrade',    '—', 15, true);
        app.addRow(rg,3,'Referable DR:','ResultReferable','—', 15, true);
        app.addRow(rg,4,'Model Score:', 'ResultScore',    '—', 13, false);
        app.addRow(rg,5,'Advice:',      'ResultReco',     '—', 12, true);
        app.ResultReco.WordWrap = 'on';
        app.ResultDetails = uilabel(rg,'Text','', ...
            'FontSize',10,'FontColor',[0.45 0.45 0.45]);
        app.ResultDetails.Layout.Row = 7;
        app.ResultDetails.Layout.Column = [1 2];
        app.ResultDetails.WordWrap = 'on';

        qual = uipanel(mid,'Title','IMAGE QUALITY (prototype thresholds)', ...
            'FontWeight','bold');
        qg = uigridlayout(qual,[6 3]);
        qg.ColumnWidth = {100,'1x',46};
        qg.RowHeight = {26,26,26,52,30,52};
        app.QualityRows = gobjects(3,2);
        dims = ["Focus","Illumination","FOV"];
        for k = 1:3
            dl = uilabel(qg,'Text',dims(k));
            dl.Layout.Row = k;
            app.QualityRows(k,1) = uilabel(qg,'Text','—');
            app.QualityRows(k,1).Layout.Row = k;
            app.QualityRows(k,1).Layout.Column = 2;
            app.QualityRows(k,2) = uilabel(qg,'Text','');
            app.QualityRows(k,2).Layout.Row = k;
            app.QualityRows(k,2).Layout.Column = 3;
        end
        ol = uilabel(qg,'Text','Overall:','FontWeight','bold');
        ol.Layout.Row = 4;
        app.QualityOverall = uilabel(qg,'Text','—','FontWeight','bold');
        app.QualityOverall.Layout.Row = 4;
        app.QualityOverall.Layout.Column = [2 3];
        el = uilabel(qg,'Text','Enhancement:','FontWeight','bold');
        el.Layout.Row = 5;
        app.QualityEnhance = uilabel(qg,'Text','—');
        app.QualityEnhance.Layout.Row = 5;
        app.QualityEnhance.Layout.Column = [2 3];
        app.RejectBanner = uilabel(qg, ...
            'Text','', 'FontWeight','bold', 'FontColor',[0.75 0.1 0.1]);
        app.RejectBanner.Layout.Row = 6;
        app.RejectBanner.Layout.Column = [1 3];
        app.QualityOverall.WordWrap = 'on';
        app.RejectBanner.WordWrap = 'on';

        % ----- Right: explainability + report -----
        right = uigridlayout(cg,[3 1]);
        right.RowHeight = {'1x',44,44};
        exp = uipanel(right,'Title','MODEL EXPLANATION (attention, not lesion proof)', ...
            'FontWeight','bold');
        eg = uigridlayout(exp,[3 1]);
        eg.RowHeight = {74,'1x',20};
        bg = uigridlayout(eg,[2 2]);
        app.ViewBtns = gobjects(4,1);
        names = ["Original","Enhanced","Grad-CAM","Overlay"];
        for k = 1:4
            app.ViewBtns(k) = uibutton(bg,'Text',names(k), ...
                'ButtonPushedFcn',@(s,e)app.showView(names(k)));
        end
        app.ExpAxes = uiaxes(eg);
        axis(app.ExpAxes,'off');
        uilabel(eg,'Text','Model attention visualization — not validated lesion localization.', ...
            'FontSize',10,'FontColor',[0.45 0.45 0.45]);
        app.ReportBtn = uibutton(right,'Text','Generate Screening Report', ...
            'Enable','off','FontWeight','bold', ...
            'BackgroundColor',[0.12 0.42 0.28],'FontColor','w', ...
            'ButtonPushedFcn',@(s,e)app.onReport());

        % ---------- Footer ----------
        ftr = uilabel(main, ...
            'Text',['Human-in-the-loop: final clinical decision by an ophthalmologist.  ' ...
            'Model R50 aptosidrid4, APTOS-test acc 82.30% · referable sens 91.93% / spec 95.08% · prototype, NOT clinically validated.'], ...
            'FontSize',11,'FontColor',[0.35 0.35 0.35], ...
            'HorizontalAlignment','center');
    end

    function addRow(app, grid, row, caption, prop, initial, fs, bold)
        cap = uilabel(grid,'Text',caption);
        cap.Layout.Row = row;
        if bold, w = 'bold'; else, w = 'normal'; end
        val = uilabel(grid,'Text',initial,'FontSize',fs,'FontWeight',w);
        val.Layout.Row = row;
        val.Layout.Column = 2;
        app.(prop) = val;
    end

    function setStatus(app, color, text)
        app.StatusLamp.Color = color;
        app.StatusLabel.Text = text;
        drawnow;
    end

    % ================= Upload =================
    function onUpload(app)
        [f,p] = uigetfile({'*.jpg;*.jpeg;*.png;*.tif','Fundus Images'}, ...
            'Select Fundus Image');
        if isequal(f,0), return; end
        try
            app.OriginalImage = imread(fullfile(p,f));
            app.ImgFileName = string(f);
            imshow(app.OriginalImage,'Parent',app.ImgAxes);
            title(app.ImgAxes, f, 'Interpreter','none');
            app.AnalyzeBtn.Enable = 'on';
            app.ReportBtn.Enable = 'off';
            app.HasResult = false;
            app.clearCards();
        catch ME
            uialert(app.Fig, ME.message, 'Upload failed');
        end
    end

    function clearCards(app)
        app.ResultSeverity.Text = '—';
        app.ResultGrade.Text = '—';
        app.ResultReferable.Text = '—';
        app.ResultReferable.FontColor = [0.2 0.2 0.2];
        app.ResultScore.Text = '—';
        app.ResultReco.Text = '—';
        app.ResultReco.BackgroundColor = 'none';
        app.ResultDetails.Text = '';
        for k = 1:3
            app.QualityRows(k,1).Text = '—';
            app.QualityRows(k,2).Text = '';
        end
        app.QualityOverall.Text = '—';
        app.QualityEnhance.Text = '—';
        app.RejectBanner.Text = '';
    end

    % ================= Analyze =================
    function onAnalyze(app)
        if isempty(app.OriginalImage), return; end
        app.setStatus([1 0.65 0], 'Analyzing…');
        drawnow;
        try
            I = app.OriginalImage;
            [app.CVal, qf] = qualityCheck(I);
            app.QFlag = string(qf);
            res = screenFundusImage(I, app.TrainedNet, 'SecondOpinion', true);
            app.Result = res;
            app.HasResult = true;
            app.updateQualityCard();
            if res.qualityStatus == "REJECT"
                app.showRejected();
            else
                app.updateResultCard();
                app.showView("Overlay");
                app.ReportBtn.Enable = 'on';
            end
            app.setStatus([0.2 0.7 0.2], 'System Ready');
        catch ME
            app.setStatus([0.2 0.7 0.2], 'System Ready');
            uialert(app.Fig, ME.message, 'Analysis failed');
        end
    end

    function updateQualityCard(app)
        res = app.Result;
        q = res.quality;
        vals = [string(sprintf('%.4f', q.FocusScore)), ...
                string(sprintf('%.3f (sd %.3f)', q.MeanIllumination, q.IlluminationStd)), ...
                string(sprintf('%.1f%%', 100*q.FOVCoverage))];
        oks = [q.FocusScore >= 0.0016, q.MeanIllumination >= 0.2340, ...
               q.FOVCoverage >= 0.475];
        for k = 1:3
            app.QualityRows(k,1).Text = vals(k);
            if oks(k)
                app.QualityRows(k,2).Text = '✓';
                app.QualityRows(k,2).FontColor = [0.1 0.55 0.1];
            else
                app.QualityRows(k,2).Text = '!';
                app.QualityRows(k,2).FontColor = [0.8 0.45 0];
            end
        end
        st = string(res.qualityStatus);
        app.QualityOverall.Text = sprintf('%s — %s', st, res.qualityReason);
        if st == "ACCEPT"
            app.QualityOverall.FontColor = [0.1 0.55 0.1];
            app.QualityEnhance.Text = 'Not needed';
        elseif st == "BORDERLINE"
            app.QualityOverall.FontColor = [0.8 0.45 0];
            app.QualityEnhance.Text = 'APPLIED (automatic)';
        else
            app.QualityOverall.FontColor = [0.75 0.1 0.1];
            app.QualityEnhance.Text = '—';
        end
    end

    function showRejected(app)
        app.RejectBanner.Text = ...
            'IMAGE NOT SUITABLE FOR ANALYSIS — please recapture the retinal image.';
        app.ResultSeverity.Text = '—';
        app.ResultGrade.Text = '—';
        app.ResultReferable.Text = 'Not assessed';
        app.ResultScore.Text = '—';
        app.ResultReco.Text = 'RECAPTURE IMAGE BEFORE SCREENING';
        app.ResultReco.BackgroundColor = [1 0.85 0.85];
        app.ReportBtn.Enable = 'off';
    end

    function updateResultCard(app)
        res = app.Result;
        app.RejectBanner.Text = '';
        app.ResultSeverity.Text = char(res.severity);
        if isfield(res,'mergedGrade') && res.mergedGrade && res.ICDRGrade == 3
            app.ResultGrade.Text = '3 (merged 3-4)';
        else
            app.ResultGrade.Text = sprintf('%d / 4', res.ICDRGrade);
        end
        if res.referableDR
            app.ResultReferable.Text = 'YES';
            app.ResultReferable.FontColor = [0.75 0.1 0.1];
        else
            app.ResultReferable.Text = 'NO';
            app.ResultReferable.FontColor = [0.1 0.55 0.1];
        end
        app.ResultScore.Text = sprintf('%.2f%%', res.modelScorePercent);
        app.ResultReco.Text = char(res.referralRecommendation);
        if res.referableDR
            app.ResultReco.BackgroundColor = [1 0.85 0.85];
        else
            app.ResultReco.BackgroundColor = [0.85 0.93 0.85];
        end
        if res.ttaUsed
            ttaTxt = sprintf('%d-view', res.ttaViews);
        else
            ttaTxt = 'off';
        end
        if res.secondOpinionConsulted
            if res.escalated
                soTxt = 'escalated to REFER';
            else
                soTxt = 'no change';
            end
        else
            soTxt = 'n/a';
        end
        app.ResultDetails.Text = sprintf( ...
            'Margin %.1fpp · %s · TTA %s · SO %s', ...
            res.scoreMargin, char(string(res.reliabilityFlag)), ...
            ttaTxt, soTxt);
    end

    % ================= Explainability views =================
    function showView(app, name)
        if ~app.HasResult
            uialert(app.Fig,'Analyze an image first.','No result yet');
            return;
        end
        allViews = ["Original","Enhanced","Grad-CAM","Overlay"];
        for b = 1:4
            if allViews(b) == string(name)
                app.ViewBtns(b).BackgroundColor = [0.08 0.34 0.60];
                app.ViewBtns(b).FontColor = 'w';
                app.ViewBtns(b).FontWeight = 'bold';
            else
                app.ViewBtns(b).BackgroundColor = [0.93 0.94 0.96];
                app.ViewBtns(b).FontColor = [0.15 0.15 0.15];
                app.ViewBtns(b).FontWeight = 'normal';
            end
        end
        res = app.Result;
        ax = app.ExpAxes;
        cla(ax);
        switch name
            case "Original"
                imshow(app.OriginalImage,'Parent',ax);
                title(ax,'Original Fundus','FontWeight','bold');
            case "Enhanced"
                imshow(res.classifierInput,'Parent',ax);
                title(ax,'Enhanced Fundus (224×224 model input)','FontWeight','bold');
            case "Grad-CAM"
                m = app.normMap(res.gradCAMMap);
                imagesc(ax, m);
                axis(ax,'image','off');
                colormap(ax,'jet'); colorbar(ax);
                if isfield(res,'gradCAMRawMax')
                    title(ax,sprintf('Grad-CAM Attention (rawmax %.3f)', ...
                        double(res.gradCAMRawMax)),'FontWeight','bold');
                else
                    title(ax,'Grad-CAM Attention','FontWeight','bold');
                end
            case "Overlay"
                if ~res.gradCAMAvailable
                    imshow(app.OriginalImage,'Parent',ax);
                    title(ax,'Overlay unavailable','FontWeight','bold');
                    return;
                end
                % Honest rendering (2026-09-25): alpha cap + caption from
                % result when present; geometry unchanged (map stretched onto
                % the same image that was classified — exact inverse).
                [H,W,~] = size(app.OriginalImage);
                m = app.normMap(imresize(res.gradCAMMap,[H W]));
                if isfield(res,'gradCAMAlphaCap')
                    alphaCap = double(res.gradCAMAlphaCap);
                else
                    alphaCap = 0.45;   % back-compat: old result structs
                end
                imshow(app.OriginalImage,'Parent',ax);
                hold(ax,'on');
                h = imagesc(ax, m);
                set(h,'AlphaData',alphaCap*m);
                axis(ax,'image','off');
                colormap(ax,'jet'); colorbar(ax);
                if isfield(res,'gradCAMCaption')
                    title(ax,sprintf('Fundus + Grad-CAM Overlay\n%s', ...
                        char(string(res.gradCAMCaption))),'FontWeight','bold');
                else
                    title(ax,'Fundus + Grad-CAM Overlay','FontWeight','bold');
                end
                hold(ax,'off');
        end
    end

    function m = normMap(~, g)
        % Same normalization as displayGradCAMOverlay.
        m = double(g);
        m = m - min(m(:));
        mx = max(m(:));
        if mx > 0, m = m ./ mx; end
    end

    % ================= Report =================
    function onReport(app)
        if ~app.HasResult, return; end
        [~,fname,~] = fileparts(app.ImgFileName);
        outPDF = fullfile(app.ProjectRoot,'reports', ...
            sprintf('screening_report_%s.pdf', fname));
        try
            [savedFile, ~] = generateScreeningPDF(outPDF, ...
                char(app.ImgFileName), char(app.QFlag), app.CVal, ...
                string(app.Result.qualityStatus), app.Result);
            uialert(app.Fig, sprintf('Report saved:\n%s', savedFile), ...
                'Report generated');
        catch ME
            uialert(app.Fig, ME.message, 'Report failed');
        end
    end
end
end
