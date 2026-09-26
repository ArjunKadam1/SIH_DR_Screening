classdef RetinaAIApp_Lesion < handle
% RETINAAIAPP_LESION - DRISHTI front end + Lesion Evidence tab (new app).
%
% Clone of RetinaAIApp for MA + P3-A vessel wiring. The old RetinaAIApp.m
% is untouched — if this new app bugs, fall back with: app = RetinaAIApp.
%
% Differences vs RetinaAIApp:
%   * Right side is a tab group: [Screening | Lesion Evidence].
%   * Lesion tab: opt-in checkbox (default OFF), FOV-mask upload button,
%     summary labels, 4 lesion views, prototype disclaimer.
%   * Lesion nets lazy-loaded on first lesion run (MA 115MB + P3-A 1.8MB):
%       results/MA_LR3E5_20260922/best_diagnostic_checkpoint.mat (bestNet)
%       results/VES_P3_NORM_20260923/arm_resize/armA_block2_iter80.mat (net)
%   * Report goes through generateScreeningPDF_Lesion (adds lesion section).
%
% Lesion labels: MA 0.42 val / P3-A vessel dFix 0.37 val, supporting
% evidence only, NOT clinical. Test-27 IDRiD untouched.
%
% Usage:
%   addpath('demo');  app = RetinaAIApp_Lesion;

properties (Access = private)
    Fig
    StatusLamp
    StatusLabel
    ImgAxes
    ExpAxes
    AnalyzeBtn
    ReportBtn
    ViewBtns            % 4x1 view buttons (screening tab)
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
    % ---- P0 UI polish (labels/cards only, no inference change) ----
    WorkflowSteps       % 6x1 workflow strip labels
    ReportStatus        % report generation status label
    ImgCaption          % filename + dims under preview
    ReasonLabel         % referral basis line (new, derived from referralRule)
    RelConf             % reliability card rows (new labels only)
    RelMargin
    RelEntropy
    RelFlag
    RelSO
    % ---- Lesion additions (new app only) ----
    ScreenTab
    LesionTab
    ReportTab           % in-app report preview tab
    ReportText          % read-only report preview area
    ReportZoom = 13     % preview font size (zoomable 10-18, view-only)
    % ---- F1 staged engine (orchestration only, no inference change) ----
    Stage = 0           % 0 idle, 1 loaded, 2 quality, 3 screened, 4 lesion,
                        % 5 reviewed, 6 reported, -1 terminal REJECT
    StageTimes = struct()  % measured per-stage seconds (quality/screen/lesion)
    LesionCheck         % opt-in checkbox
    FOVButton
    FOVLabel
    LesionViewBtns      % 4x1 lesion view buttons
    LesionAxes
    LesionSummary
    LesionNoteLabel
    LesionView = "MA overlay"
    MANet = []
    VesNet = []
    LesionNetsReady = false
    FOVMask = []
    FOVFileName = ""
end

methods (Access = public)
    function app = RetinaAIApp_Lesion()
        % Derive project root from this file's location (works on any
        % machine / MATLAB Drive checkout, mirrors runDRDemoV2).
        app.ProjectRoot = fileparts(fileparts(mfilename('fullpath')));
        addpath(fullfile(app.ProjectRoot,'preprocessing'));
        addpath(fullfile(app.ProjectRoot,'quality'));
        addpath(fullfile(app.ProjectRoot,'demo'));
        addpath(fullfile(app.ProjectRoot,'classification'));
        addpath(fullfile(app.ProjectRoot,'segmentation','vessels'));

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

    function out = headlessAnalyze(app, imgPath, runLesion)
        % HEADLESSANALYZE - regression hook: loads image, runs the same
        % onAnalyze path as the Analyze button, returns visible card texts.
        % runLesion (default false): false = opt OUT of lesion, preserving
        % the screening-only parity path; true = exercise the auto path.
        if nargin < 3 || isempty(runLesion)
            runLesion = false;
        end
        app.OriginalImage = imread(imgPath);
        [~,f,e] = fileparts(imgPath);
        app.ImgFileName = string([f e]);
        app.AnalyzeBtn.Enable = 'on';
        app.LesionCheck.Value = ~runLesion;  % checkbox is opt-OUT (checked = skip)
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
            'lesion',char(app.LesionSummary.Text), ...
            'lesionNote',char(app.LesionNoteLabel.Text), ...
            'reportEnabled',char(app.ReportBtn.Enable));
    end
end

methods (Access = private)

    function buildUI(app)
        % Centered fixed window: same 1360x820 on any laptop screen.
        scr = get(groot,'ScreenSize');
        winXY = [max(0,(scr(3)-1360)/2), max(0,(scr(4)-820)/2)];
        app.Fig = uifigure('Name','DRISHTI - DR Screening Assistant + Lesion', ...
            'Position',[winXY 1360 820], 'Color',[0.92 0.94 0.96]);
        main = uigridlayout(app.Fig,[4 1]);
        main.RowHeight = {70,28,'1x',64};
        main.Padding = [10 10 10 10];

        % ---------- Header ----------
        navy = [0.07 0.18 0.32];
        hdr = uipanel(main,'BackgroundColor',navy,'BorderType','none');
        hg = uigridlayout(hdr,[2 2]);
        hg.ColumnWidth = {'1x',210};
        hg.RowHeight = {'1.2x','1x'};
        hg.BackgroundColor = navy;
        hg.Padding = [14 4 12 4];
        t = uilabel(hg,'Text','DRISHTI + Lesion', ...
            'FontSize',23,'FontWeight','bold','FontColor','w');
        t.Layout.Row = 1; t.Layout.Column = 1;
        subt = uilabel(hg, ...
            'Text','Explainable AI for Diabetic Retinopathy Screening · Prototype — not clinically validated', ...
            'FontSize',12,'FontColor',[0.72 0.81 0.92]);
        subt.Layout.Row = 2; subt.Layout.Column = 1;
        sg = uigridlayout(hg,[1 2]);
        sg.ColumnWidth = {24,'1x'};
        sg.BackgroundColor = navy;
        sg.Layout.Row = [1 2]; sg.Layout.Column = 2;
        app.StatusLamp = uilamp(sg,'Color',[0.2 0.7 0.2]);
        app.StatusLabel = uilabel(sg,'Text','System Ready', ...
            'FontColor','w','FontSize',13,'FontWeight','bold');

        % ---------- Workflow strip (UI-only, reflects real app state) ----------
        wstrip = uipanel(main,'BorderType','none','BackgroundColor',[0.92 0.94 0.96]);
        wg = uigridlayout(wstrip,[1 6]);
        wg.Padding = [4 2 4 2];
        wnames = ["Image","Quality","AI Screening","Reliability","Referral","Report"];
        app.WorkflowSteps = gobjects(6,1);
        for wk = 1:6
            app.WorkflowSteps(wk) = uilabel(wg, ...
                'Text',sprintf('○ %d. %s',wk,wnames(wk)), ...
                'FontSize',11,'FontColor',[0.55 0.55 0.55], ...
                'HorizontalAlignment','center');
            app.WorkflowSteps(wk).Layout.Column = wk;
        end

        % ---------- Content: 3 columns ----------
        cg = uigridlayout(main,[1 3]);
        cg.ColumnWidth = {410,395,'1x'};
        cg.Padding = [0 6 0 6];

        % ----- Left: image -----
        left = uipanel(cg,'Title','FUNDUS IMAGE','FontWeight','bold', ...
            'FontSize',14,'ForegroundColor',[0.07 0.18 0.32]);
        lg = uigridlayout(left,[6 1]);
        lg.RowHeight = {'1x',20,38,44,38,8};
        app.ImgAxes = uiaxes(lg);
        axis(app.ImgAxes,'off');
        title(app.ImgAxes,'No image loaded','Color',[0.5 0.5 0.5]);
        app.ImgCaption = uilabel(lg,'Text','No image loaded.', ...
            'FontSize',11,'FontColor',[0.45 0.45 0.45]);
        app.ImgCaption.Layout.Row = 2;
        app.ImgCaption.WordWrap = 'on';
        uibutton(lg,'Text','Upload Fundus Image','FontSize',13, ...
            'ButtonPushedFcn',@(s,e)app.onUpload());
        app.AnalyzeBtn = uibutton(lg,'Text','ANALYZE IMAGE', ...
            'FontSize',14,'FontWeight','bold','Enable','off', ...
            'BackgroundColor',[0.05 0.38 0.62],'FontColor','w', ...
            'ButtonPushedFcn',@(s,e)app.onAnalyze());
        app.AnalyzeBtn.Layout.Row = 4;
        nb = uibutton(lg,'Text','NEW SCREENING / CLEAR', ...
            'FontWeight','bold','BackgroundColor',[0.42 0.42 0.44],'FontColor','w', ...
            'ButtonPushedFcn',@(s,e)app.onNewScreening());
        nb.Layout.Row = 5;

        % ----- Middle: result + reliability + quality -----
        mid = uigridlayout(cg,[3 1]);
        mid.RowHeight = {'1.12x','1x','0.66x'};
        res = uipanel(mid,'Title','AI SCREENING RESULT','FontWeight','bold', ...
            'FontSize',14,'ForegroundColor',[0.07 0.18 0.32]);
        rg = uigridlayout(res,[8 2]);
        rg.ColumnWidth = {132,'1x'};
        rg.RowHeight = {32,30,30,30,52,24,4,36};
        app.addRow(rg,1,'DR Severity:', 'ResultSeverity', '—', 20, true);
        app.addRow(rg,2,'ICDR Grade:',  'ResultGrade',    '—', 16, true);
        app.addRow(rg,3,'Referable DR:','ResultReferable','—', 16, true);
        app.addRow(rg,4,'Model Score:', 'ResultScore',    '—', 14, false);
        app.addRow(rg,5,'Advice:',      'ResultReco',     '—', 14, true);
        app.ResultReco.WordWrap = 'on';
        % NEW label (referral basis, derived from referralRule — text only).
        app.ReasonLabel = uilabel(rg,'Text','', ...
            'FontSize',12,'FontAngle','italic','FontColor',[0.3 0.3 0.3]);
        app.ReasonLabel.Layout.Row = 6;
        app.ReasonLabel.Layout.Column = [1 2];
        app.ReasonLabel.WordWrap = 'on';
        app.ResultDetails = uilabel(rg,'Text','', ...
            'FontSize',11,'FontColor',[0.45 0.45 0.45]);
        app.ResultDetails.Layout.Row = 8;
        app.ResultDetails.Layout.Column = [1 2];
        app.ResultDetails.WordWrap = 'on';

        qual = uipanel(mid,'Title','IMAGE QUALITY (prototype thresholds)', ...
            'FontWeight','bold','FontSize',14,'ForegroundColor',[0.07 0.18 0.32]);
        qg = uigridlayout(qual,[6 3]);
        qg.ColumnWidth = {110,'1x',48};
        qg.RowHeight = {28,28,28,54,32,54};
        app.QualityRows = gobjects(3,2);
        dims = ["Focus","Illumination","FOV"];
        for k = 1:3
            dl = uilabel(qg,'Text',dims(k),'FontSize',12);
            dl.Layout.Row = k;
            app.QualityRows(k,1) = uilabel(qg,'Text','—','FontSize',13);
            app.QualityRows(k,1).Layout.Row = k;
            app.QualityRows(k,1).Layout.Column = 2;
            app.QualityRows(k,2) = uilabel(qg,'Text','','FontSize',13,'FontWeight','bold');
            app.QualityRows(k,2).Layout.Row = k;
            app.QualityRows(k,2).Layout.Column = 3;
        end
        ol = uilabel(qg,'Text','Overall:','FontWeight','bold','FontSize',12);
        ol.Layout.Row = 4;
        app.QualityOverall = uilabel(qg,'Text','—','FontWeight','bold','FontSize',14);
        app.QualityOverall.Layout.Row = 4;
        app.QualityOverall.Layout.Column = [2 3];
        el = uilabel(qg,'Text','Enhancement:','FontWeight','bold','FontSize',12);
        el.Layout.Row = 5;
        app.QualityEnhance = uilabel(qg,'Text','—','FontSize',12);
        app.QualityEnhance.Layout.Row = 5;
        app.QualityEnhance.Layout.Column = [2 3];
        app.RejectBanner = uilabel(qg, ...
            'Text','', 'FontWeight','bold', 'FontColor',[0.75 0.1 0.1]);
        app.RejectBanner.Layout.Row = 6;
        app.RejectBanner.Layout.Column = [1 3];
        app.QualityOverall.WordWrap = 'on';
        app.RejectBanner.WordWrap = 'on';

        % ----- Reliability card (NEW labels only, values from Result) -----
        relp = uipanel(mid,'Title','MODEL RELIABILITY','FontWeight','bold', ...
            'FontSize',14,'ForegroundColor',[0.07 0.18 0.32]);
        rlg = uigridlayout(relp,[5 2]);
        rlg.ColumnWidth = {132,'1x'};
        rlg.RowHeight = {26,26,26,26,26};
        app.addRow(rlg,1,'Confidence:', 'RelConf',   '—', 13, false);
        app.addRow(rlg,2,'Margin:',     'RelMargin', '—', 13, false);
        app.addRow(rlg,3,'Entropy:',    'RelEntropy','—', 13, false);
        app.addRow(rlg,4,'Reliability:','RelFlag',   '—', 13, true);
        app.addRow(rlg,5,'2nd opinion:','RelSO',     '—', 13, false);

        % ----- Right: TAB GROUP (new) -----
        right = uigridlayout(cg,[1 1]);
        tg = uitabgroup(right);

        % ===== Tab 1: Screening (same as old right panel) =====
        app.ScreenTab = uitab(tg,'Title','Screening');
        sgrid = uigridlayout(app.ScreenTab,[1 1]);
        exp = uipanel(sgrid,'Title','MODEL EXPLANATION (attention, not lesion proof)', ...
            'FontWeight','bold','FontSize',14,'ForegroundColor',[0.07 0.18 0.32]);
        eg = uigridlayout(exp,[3 1]);
        eg.RowHeight = {74,'1x',20};
        bg = uigridlayout(eg,[2 2]);
        app.ViewBtns = gobjects(4,1);
        names = ["Original","Enhanced","Grad-CAM","Overlay"];
        for k = 1:4
            app.ViewBtns(k) = uibutton(bg,'Text',names(k),'FontSize',12, ...
                'ButtonPushedFcn',@(s,e)app.showView(names(k)));
        end
        app.ExpAxes = uiaxes(eg);
        axis(app.ExpAxes,'off');
        uilabel(eg,'Text','Model attention visualization — not validated lesion localization.', ...
            'FontSize',10,'FontColor',[0.45 0.45 0.45]);

        % ===== Tab 2: Lesion Evidence (NEW) =====
        app.LesionTab = uitab(tg,'Title','Lesion Evidence');
        lgrid = uigridlayout(app.LesionTab,[4 1]);
        lgrid.RowHeight = {106,40,'1x',40};
        opt = uipanel(lgrid,'Title','LESION EVIDENCE (prototype, opt-in)', ...
            'FontWeight','bold','FontSize',14,'ForegroundColor',[0.07 0.18 0.32]);
        og = uigridlayout(opt,[4 2]);
        og.ColumnWidth = {'1x',150};
        og.RowHeight = {26,26,24,20};
        app.LesionCheck = uicheckbox(og,'Text','Skip lesion evidence (faster screening)', ...
            'Value',false,'FontWeight','bold','FontSize',12);  % unchecked = RUN (auto)
        app.LesionCheck.Layout.Row = 1;
        app.LesionCheck.Layout.Column = [1 2];
        app.FOVButton = uibutton(og,'Text','Upload FOV mask','FontSize',12, ...
            'ButtonPushedFcn',@(s,e)app.onUploadFOV());
        app.FOVButton.Layout.Row = 2;
        app.FOVButton.Layout.Column = 1;
        app.FOVLabel = uilabel(og,'Text','No FOV mask — vessel % is full-frame.', ...
            'FontSize',10,'FontColor',[0.5 0.5 0.5]);
        app.FOVLabel.Layout.Row = 2;
        app.FOVLabel.Layout.Column = 2;
        app.FOVLabel.WordWrap = 'on';
        app.LesionSummary = uilabel(og,'Text','Lesion not run yet.', ...
            'FontSize',12,'FontWeight','bold');
        app.LesionSummary.Layout.Row = 3;
        app.LesionSummary.Layout.Column = [1 2];
        app.LesionSummary.WordWrap = 'on';
        % Static honesty caption (never overwritten by results).
        mac = uilabel(og,'Text','MA = early-stage dots; sparse output is expected and the count is not a severity grade.', ...
            'FontSize',10,'FontAngle','italic','FontColor',[0.45 0.45 0.45]);
        mac.Layout.Row = 4;
        mac.Layout.Column = [1 2];
        mac.WordWrap = 'on';

        lbtn = uigridlayout(lgrid,[1 4]);
        app.LesionViewBtns = gobjects(4,1);
        lnames = ["MA overlay","Vessel overlay","MA heat","Vessel heat"];
        for k = 1:4
            app.LesionViewBtns(k) = uibutton(lbtn,'Text',lnames(k),'FontSize',12, ...
                'ButtonPushedFcn',@(s,e)app.showLesionView(lnames(k)));
        end
        app.LesionAxes = uiaxes(lgrid);
        axis(app.LesionAxes,'off');
        title(app.LesionAxes,'Analyze with lesion checkbox on','Color',[0.5 0.5 0.5]);
        app.LesionNoteLabel = uilabel(lgrid, ...
            'Text','MA 0.42 val / P3-A vessel dFix 0.37 val. Supporting evidence only, NOT clinical.', ...
            'FontSize',10,'FontColor',[0.45 0.45 0.45]);
        app.LesionNoteLabel.WordWrap = 'on';

        % ===== Tab 3: Report (in-app preview + PDF export) =====
        % Preview text comes from the SAME shared builder as the PDF, so
        % the two can never drift apart.
        app.ReportTab = uitab(tg,'Title','Report');
        rpgrid = uigridlayout(app.ReportTab,[3 1]);
        rpgrid.RowHeight = {'1x',44,26};
        app.ReportText = uitextarea(rpgrid,'Value','Analyze an image to preview the report.', ...
            'Editable','off','FontName','Courier New','FontSize',13);
        % Button row nests Generate + zoom so tab borders never move.
        brow = uigridlayout(rpgrid,[1 3]);
        brow.ColumnWidth = {'1x',52,52};
        brow.Padding = [0 0 0 0];
        brow.Layout.Row = 2;
        app.ReportBtn = uibutton(brow,'Text','Generate Screening Report (PDF)', ...
            'Enable','off','FontWeight','bold','FontSize',13, ...
            'BackgroundColor',[0.12 0.42 0.28],'FontColor','w', ...
            'ButtonPushedFcn',@(s,e)app.onReport());
        app.ReportBtn.Layout.Column = 1;
        zout = uibutton(brow,'Text','A-','FontSize',13,'FontWeight','bold', ...
            'Tooltip','Smaller preview text', ...
            'ButtonPushedFcn',@(s,e)app.zoomReport(-1));
        zout.Layout.Column = 2;
        zin = uibutton(brow,'Text','A+','FontSize',13,'FontWeight','bold', ...
            'Tooltip','Larger preview text', ...
            'ButtonPushedFcn',@(s,e)app.zoomReport(1));
        zin.Layout.Column = 3;
        app.ReportStatus = uilabel(rpgrid,'Text','', ...
            'FontSize',10,'FontColor',[0.1 0.55 0.1]);
        app.ReportStatus.Layout.Row = 3;
        app.ReportStatus.WordWrap = 'on';

        % ---------- Footer (metrics traced: HandheldDR/reports/all_models_report.txt,
        % APTOS internal test n=548, 13-Sep-2026 — P0.0 CLOSED) ----------
        ftr = uilabel(main, ...
            'Text',['Human-in-the-loop: final clinical decision by an ophthalmologist.  ' ...
            'Model R50 aptosidrid4, APTOS-test acc 82.30% · referable sens 91.93% / spec 95.08% · prototype, NOT clinically validated. ' ...
            'Source: HandheldDR/reports/all_models_report.txt (APTOS-test n=548, 13-Sep-2026).'], ...
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
            app.ImgCaption.Text = sprintf('%s (%d×%d)', f, ...
                size(app.OriginalImage,1), size(app.OriginalImage,2));
            app.AnalyzeBtn.Enable = 'on';
            app.ReportBtn.Enable = 'off';
            app.HasResult = false;
            app.clearCards();
            app.setStage(1);
        catch ME
            fprintf('UPLOAD ERROR: %s\n', ME.message);
            uialert(app.Fig, ...
                'Unable to read this image. Please select a valid JPG, PNG, or TIFF fundus image.', ...
                'Upload failed');
        end
    end

    function onUploadFOV(app)
        [f,p] = uigetfile({'*.jpg;*.jpeg;*.png;*.tif;*.gif;*.bmp','FOV Masks'}, ...
            'Select FOV Mask (optional, vessel only)');
        if isequal(f,0), return; end
        try
            m = imread(fullfile(p,f));
            if ndims(m) == 3
                m = m(:,:,1);
            end
            app.FOVMask = logical(m > 0);
            app.FOVFileName = string(f);
            app.FOVLabel.Text = sprintf('FOV: %s (%d px)', f, nnz(app.FOVMask));
            app.FOVLabel.FontColor = [0.1 0.55 0.1];
        catch ME
            fprintf('FOV UPLOAD ERROR: %s\n', ME.message);
            uialert(app.Fig, ...
                'Unable to read this mask file. Please select a valid image file (white = inside FOV).', ...
                'FOV upload failed');
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
        app.ReasonLabel.Text = '';
        for k = 1:3
            app.QualityRows(k,1).Text = '—';
            app.QualityRows(k,2).Text = '';
        end
        app.QualityOverall.Text = '—';
        app.QualityEnhance.Text = '—';
        app.RejectBanner.Text = '';
        app.RelConf.Text = '—';
        app.RelMargin.Text = '—';
        app.RelEntropy.Text = '—';
        app.RelFlag.Text = '—';
        app.RelFlag.FontColor = [0.2 0.2 0.2];
        app.RelSO.Text = '—';
        app.ReportStatus.Text = '';
        app.ReportText.Value = 'Analyze an image to preview the report.';
        app.ReportZoom = 13;
        app.ReportText.FontSize = 13;
        app.StageTimes = struct();
        app.setStage(0);
        app.clearLesion();
    end

    function onNewScreening(app)
        % NEW SCREENING / CLEAR — resets case, keeps expensive nets loaded.
        app.OriginalImage = [];
        app.ImgFileName = "";
        app.Result = struct();
        app.HasResult = false;
        app.CVal = NaN;
        app.QFlag = "";
        app.FOVMask = [];
        app.FOVFileName = "";
        app.FOVLabel.Text = 'No FOV mask — vessel % is full-frame.';
        app.FOVLabel.FontColor = [0.5 0.5 0.5];
        app.ImgCaption.Text = 'No image loaded.';
        cla(app.ImgAxes);
        title(app.ImgAxes,'No image loaded','Color',[0.5 0.5 0.5]);
        cla(app.ExpAxes);
        app.AnalyzeBtn.Enable = 'off';
        app.ReportBtn.Enable = 'off';
        app.clearCards();
        app.setStatus([0.2 0.7 0.2], 'System Ready');
    end

    function setStage(app, s)
        % F1 engine: record pipeline state + mirror to workflow strip.
        % The strip has no lesion/review steps, so S3-S5 all show level 5;
        % terminal REJECT (-1) shows quality assessed, rest pending.
        app.Stage = s;
        if s < 0
            app.updateWorkflow(2);
        else
            map = [0 1 2 5 5 5 6];
            app.updateWorkflow(map(min(s+1,7)));
        end
    end

    function updateWorkflow(app, level)
        % Workflow strip — UI-only mirror of real app state (no timers).
        names = ["Image","Quality","AI Screening","Reliability","Referral","Report"];
        for k = 1:6
            if k <= level
                app.WorkflowSteps(k).Text = sprintf('✓ %d. %s',k,names(k));
                app.WorkflowSteps(k).FontColor = [0.1 0.55 0.1];
                app.WorkflowSteps(k).FontWeight = 'bold';
            else
                app.WorkflowSteps(k).Text = sprintf('○ %d. %s',k,names(k));
                app.WorkflowSteps(k).FontColor = [0.55 0.55 0.55];
                app.WorkflowSteps(k).FontWeight = 'normal';
            end
        end
    end

    function updateReliabilityCard(app)
        % Reliability card — new labels only, values straight from Result.
        res = app.Result;
        if isfinite(res.calScorePercent)
            app.RelConf.Text = sprintf('%.2f%% (cal %.2f%%)', ...
                res.modelScorePercent, res.calScorePercent);
        else
            app.RelConf.Text = sprintf('%.2f%% (cal n/a)', res.modelScorePercent);
        end
        app.RelMargin.Text = sprintf('%.1f pp', res.scoreMargin);
        app.RelEntropy.Text = sprintf('%.3f', res.scoreEntropy);
        app.RelFlag.Text = char(string(res.reliabilityFlag));
        switch char(string(res.reliabilityFlag))
            case 'high'
                app.RelFlag.FontColor = [0.1 0.55 0.1];
            case 'borderline'
                app.RelFlag.FontColor = [0.8 0.45 0];
            otherwise
                app.RelFlag.FontColor = [0.75 0.1 0.1];
        end
        if res.secondOpinionConsulted
            if res.escalated
                app.RelSO.Text = 'Escalated to REFER';
            else
                app.RelSO.Text = 'Consulted — no change';
            end
        else
            app.RelSO.Text = 'Not consulted';
        end
    end

    function clearLesion(app)
        app.LesionSummary.Text = 'Lesion not run yet.';
        app.LesionNoteLabel.Text = ...
            'MA 0.42 val / P3-A vessel dFix 0.37 val. Supporting evidence only, NOT clinical.';
        cla(app.LesionAxes);
        title(app.LesionAxes,'Analyze with lesion checkbox on','Color',[0.5 0.5 0.5]);
    end

    function ensureLesionNets(app)
        % Lazy-load MA (115MB) + P3-A (1.8MB) once, on first lesion run.
        if app.LesionNetsReady, return; end
        maPath = fullfile(app.ProjectRoot,'results','MA_LR3E5_20260922', ...
            'best_diagnostic_checkpoint.mat');
        vesPath = fullfile(app.ProjectRoot,'results','VES_P3_NORM_20260923', ...
            'arm_resize','armA_block2_iter80.mat');
        S1 = load(maPath,'bestNet');
        app.MANet = S1.bestNet;
        S2 = load(vesPath,'net');
        app.VesNet = S2.net;
        app.LesionNetsReady = true;
    end

    % ================= Analyze =================
    function onAnalyze(app)
        if isempty(app.OriginalImage), return; end
        app.setStatus([1 0.65 0], 'Analyzing…');
        drawnow;
        try
            I = app.OriginalImage;
            tStage = tic;  % F1: measured per-stage seconds (honest timings)
            [app.CVal, qf] = qualityCheck(I);
            app.QFlag = string(qf);
            res = screenFundusImage(I, app.TrainedNet, 'SecondOpinion', true);
            app.Result = res;
            app.HasResult = true;
            app.updateQualityCard();
            app.StageTimes.quality = toc(tStage);
            if res.qualityStatus == "REJECT"
                app.showRejected();
            else
                app.updateResultCard();
                app.updateReliabilityCard();
                app.StageTimes.screen = toc(tStage);
                app.showView("Overlay");
                app.ReportBtn.Enable = 'on';
            end
            % ---- Lesion evidence (AUTO unless opted out — F1) ----
            % Never breaks screening: failures land in lesionNote.
            app.clearLesion();
            if ~app.LesionCheck.Value && res.qualityStatus ~= "REJECT"
                try
                    app.setStatus([1 0.65 0], 'Analyzing lesions…');
                    drawnow;
                    app.ensureLesionNets();
                    % Resize FOV mask to current image if sizes differ.
                    fov = app.FOVMask;
                    if ~isempty(fov) && ~isequal(size(fov), [size(I,1) size(I,2)])
                        fov = imresize(fov, [size(I,1) size(I,2)], 'nearest');
                    end
                    opts = struct('runMA', true, 'runVessel', true, ...
                        'maThr', 0.5, 'vesMode', 'percentile', ...
                        'vesThr', 0.5, 'vesTargetFrac', 0.09, 'vesMinArea', 20);
                    Lc = runLesionEvidence(I, app.MANet, app.VesNet, fov, opts);
                    fn = fieldnames(Lc);
                    for fi = 1:numel(fn)
                        app.Result.(fn{fi}) = Lc.(fn{fi});
                    end
                    app.updateLesionCard();
                    app.showLesionView("MA overlay");
                catch ME
                    app.Result.maAvailable = false;
                    app.Result.vesselAvailable = false;
                    app.Result.lesionNote = sprintf('Lesion failed: %s', ME.message);
                    app.updateLesionCard();
                end
            end
            if res.qualityStatus ~= "REJECT"
                app.StageTimes.lesion = toc(tStage);
                app.setStage(4);  % lesion attempted (or skipped): evidence stage done
            end
            app.renderReportPreview();
            if res.qualityStatus ~= "REJECT"
                app.setStage(5);  % reviewed: all cards + views + preview rendered
            end
            app.setStatus([0.2 0.7 0.2], 'System Ready');
        catch ME
            app.setStatus([0.2 0.7 0.2], 'System Ready');
            fprintf('ANALYZE ERROR: %s\n', ME.message);
            uialert(app.Fig, ...
                'Screening analysis could not be completed. Technical details have been logged.', ...
                'Analysis failed');
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
        app.setStage(2);
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
        app.RelConf.Text = 'Not assessed';
        app.RelMargin.Text = 'Not assessed';
        app.RelEntropy.Text = 'Not assessed';
        app.RelFlag.Text = 'Not assessed';
        app.RelSO.Text = 'Not assessed';
        app.ReportBtn.Enable = 'off';
        app.setStage(-1);  % terminal REJECT: strip stays at Quality
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
        % Referral basis line (new label; derived from referralRule only).
        if res.referableDR
            if isfield(res,'referralRule') && res.referralRule == "referScore"
                basis = 'Reason: model probability rule ≥ 0.30';
            else
                basis = 'Reason: ICDR Grade ≥ 2';
            end
            if isfield(res,'escalated') && res.escalated
                basis = basis + ' + second-opinion escalation';
            end
        else
            basis = 'Reason: below referral threshold — routine screening';
        end
        app.ReasonLabel.Text = char(basis);
        app.setStage(3);
    end

    function renderReportPreview(app)
        % In-app preview — same shared builder as the PDF export.
        % Preview failure must never break cards (same discipline as lesion).
        try
            if ~app.HasResult
                app.ReportText.Value = 'Analyze an image to preview the report.';
                return;
            end
            app.ReportText.Value = buildReportLinesLesion( ...
                char(app.ImgFileName), char(app.QFlag), app.CVal, ...
                string(app.Result.qualityStatus), app.Result);
        catch ME
            fprintf('PREVIEW ERROR: %s\n', ME.message);
            app.ReportText.Value = ...
                'Report preview unavailable for this state. PDF export may still work.';
        end
    end

    function zoomReport(app, delta)
        % Preview zoom — view-only, clamped 10-18pt. Never touches results.
        app.ReportZoom = min(18, max(10, app.ReportZoom + delta));
        app.ReportText.FontSize = app.ReportZoom;
    end

    function updateLesionCard(app)
        res = app.Result;
        parts = {};
        if isfield(res,'maAvailable') && res.maAvailable
            parts{end+1} = sprintf('MA dots: %d @%.2f (%.1fs)', ... %#ok<AGROW>
                res.maCount, res.maThr, res.maTimeSec);
        elseif isfield(res,'maAvailable')
            parts{end+1} = 'MA: failed/skipped'; %#ok<AGROW>
        else
            parts{end+1} = 'MA: not run'; %#ok<AGROW>
        end
        if isfield(res,'vesselAvailable') && res.vesselAvailable
            parts{end+1} = sprintf('Vessel: %.2f%%FOV @%.2f (%.1fs)', ... %#ok<AGROW>
                res.vesselFracFOV, res.vesselThr, res.vesselTimeSec);
        elseif isfield(res,'vesselAvailable')
            parts{end+1} = 'Vessel: no FOV mask / failed'; %#ok<AGROW>
        else
            parts{end+1} = 'Vessel: not run'; %#ok<AGROW>
        end
        app.LesionSummary.Text = strjoin(parts, ' · ');
        if isfield(res,'lesionNote')
            app.LesionNoteLabel.Text = char(string(res.lesionNote));
        end
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
        colorbar(ax,'off');  % drop stale bars when switching views
        switch name
            case "Original"
                imshow(app.OriginalImage,'Parent',ax);
                title(ax,'Original Fundus','FontWeight','bold');
            case "Enhanced"
                imshow(res.classifierInput,'Parent',ax);
                title(ax,'Enhanced Fundus (224×224 model input)','FontWeight','bold');
            case "Grad-CAM"
                % Flat-map guard (UI-only): on saturated predictions the
                % gradCAM map can be all zeros (rawmax 0.000). Showing it
                % raw renders a misleading solid-green field (axes autoscale
                % to [-1 1]). Show the original with an honest caption instead.
                m0 = double(res.gradCAMMap);
                if max(m0(:)) - min(m0(:)) <= 1e-6
                    imshow(app.OriginalImage,'Parent',ax);
                    title(ax,['Grad-CAM flat (rawmax 0) — fully confident prediction,' ...
                        ' no focus region. See Overlay view.'],'FontWeight','bold');
                    return;
                end
                m = app.normMap(res.gradCAMMap);
                imagesc(ax, m);
                clim(ax,[0 1]);
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

    function showLesionView(app, name)
        app.LesionView = string(name);
        allViews = ["MA overlay","Vessel overlay","MA heat","Vessel heat"];
        for b = 1:4
            if allViews(b) == string(name)
                app.LesionViewBtns(b).BackgroundColor = [0.08 0.34 0.60];
                app.LesionViewBtns(b).FontColor = 'w';
                app.LesionViewBtns(b).FontWeight = 'bold';
            else
                app.LesionViewBtns(b).BackgroundColor = [0.93 0.94 0.96];
                app.LesionViewBtns(b).FontColor = [0.15 0.15 0.15];
                app.LesionViewBtns(b).FontWeight = 'normal';
            end
        end
        if ~app.HasResult
            uialert(app.Fig,'Analyze an image first.','No result yet');
            return;
        end
        res = app.Result;
        ax = app.LesionAxes;
        cla(ax);
        colorbar(ax,'off');  % drop stale bars when switching views
        I = app.OriginalImage;
        switch string(name)
            case "MA overlay"
                if ~(isfield(res,'maAvailable') && res.maAvailable)
                    imshow(I,'Parent',ax);
                    title(ax,'MA not run — tick checkbox + Analyze','FontWeight','bold');
                    return;
                end
                imshow(labeloverlay(I, res.maMask, 'Transparency', 0.55), 'Parent', ax);
                title(ax,sprintf('MA dots: %d @%.2f (prototype)', ...
                    res.maCount, res.maThr),'FontWeight','bold');
            case "Vessel overlay"
                if ~(isfield(res,'vesselAvailable') && res.vesselAvailable)
                    imshow(I,'Parent',ax);
                    title(ax,'Vessel not run — tick checkbox + Analyze','FontWeight','bold');
                    return;
                end
                imshow(labeloverlay(I, res.vesselMask, 'Transparency', 0.55), 'Parent', ax);
                title(ax,sprintf('P3-A vessel %.2f%% @%.2f (prototype)', ...
                    res.vesselFracFOV, res.vesselThr),'FontWeight','bold');
            case "MA heat"
                if ~(isfield(res,'maAvailable') && res.maAvailable)
                    imshow(I,'Parent',ax);
                    title(ax,'MA not run — tick checkbox + Analyze','FontWeight','bold');
                    return;
                end
                m = app.normMap(res.maProbMap);
                imagesc(ax, m);
                axis(ax,'image','off');
                colormap(ax,'hot'); colorbar(ax);
                title(ax,'MA probability','FontWeight','bold');
            case "Vessel heat"
                if ~(isfield(res,'vesselAvailable') && res.vesselAvailable)
                    imshow(I,'Parent',ax);
                    title(ax,'Vessel not run — tick checkbox + Analyze','FontWeight','bold');
                    return;
                end
                m = app.normMap(res.vesselProbMap);
                imagesc(ax, m);
                axis(ax,'image','off');
                colormap(ax,'hot'); colorbar(ax);
                title(ax,'P3-A vessel probability','FontWeight','bold');
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
            sprintf('screening_report_%s_lesion.pdf', fname));
        try
            [savedFile, ~] = generateScreeningPDF_Lesion(outPDF, ...
                char(app.ImgFileName), char(app.QFlag), app.CVal, ...
                string(app.Result.qualityStatus), app.Result);
            [~,sname,sext] = fileparts(savedFile);
            app.ReportStatus.Text = sprintf('Report saved: %s%s', sname, sext);
            app.ReportStatus.FontColor = [0.1 0.55 0.1];
            app.setStage(6);
            uialert(app.Fig, sprintf('Report saved:\n%s', savedFile), ...
                'Report generated');
        catch ME
            fprintf('REPORT ERROR: %s\n', ME.message);
            app.ReportStatus.Text = 'Report generation failed.';
            app.ReportStatus.FontColor = [0.75 0.1 0.1];
            uialert(app.Fig, ...
                'The screening report could not be generated. Technical details have been logged.', ...
                'Report failed');
        end
    end
end
end
