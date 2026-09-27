# DRISHTI — Explainable AI for Diabetic Retinopathy Screening (SIH 2026 prototype)

MATLAB screening workstation: upload → quality gate → R50 grading + referable
decision + Grad-CAM → optional MA / vessel lesion evidence → screening report.

## Which app to run (for evaluators)

RUN THIS — `demo/RetinaAIApp_Lesion.m` (new):
```matlab
addpath('demo'); app = RetinaAIApp_Lesion;
```
Full workstation: screening + Grad-CAM + MA/vessel lesion evidence + in-app
Report tab + PDF export. This is the app in the demo video.

NOT this (unless the new app fails to launch):
`demo/RetinaAIApp.m` — previous classifier-only version, kept purely as a
fallback. Same pipeline, no lesion tab, no report tab.

## Prerequisites (software only)

MATLAB app (`RetinaAIApp_Lesion`) — required:
- MATLAB R2026a Desktop
- Deep Learning Toolbox (`classify`, `forward`, `dlnetwork`, `dlarray`, `gradCAM`)
- Image Processing Toolbox (`imresize`, `labeloverlay`, `bwareaopen`, `bwconncomp`, morphology funcs)

MATLAB app — optional (graceful fallbacks, nothing breaks without them):
- MATLAB Report Generator — without it, reports auto-fall back to figure-PDF, then `.txt`
- GPU — opportunistic only; the app runs fully on CPU with no extra toolbox

Explicitly NOT needed for the MATLAB app: Signal Processing, Statistics and
Machine Learning (legacy scripts only), Simulink/SimEvents (capacity models
only), MATLAB Compiler (exe route only).

Check yours in the MATLAB Command Window with:
```matlab
ver
```
Confirm `Deep Learning Toolbox` and `Image Processing Toolbox` appear in the list.

## Quick start
1. Download the 3 model files — see **[MODELS.md](MODELS.md)** (GitHub Release `v0.1-prototype`).
2. `addpath('demo'); app = RetinaAIApp_Lesion;` (previous `RetinaAIApp` kept as fallback).
3. Upload a **raw** fundus photo → ANALYZE → Lesion tab → Report tab → Generate PDF.

## Map
- `demo/` — app (`RetinaAIApp_Lesion.m`), pipeline (`screenFundusImage.m`), lesion helper (`runLesionEvidence.m`), report (`generateScreeningPDF_Lesion.m` + shared `buildReportLinesLesion.m`), smoke scripts.
- `classification/`, `preprocessing/`, `quality/` — grading, enhancement, quality gates.
- `segmentation/microaneurysms/`, `segmentation/vessels/training_VES_P2B/` — MA + vessel training (frozen).
- `simulink/` — DRISHTI workflow/telemedicine models (video: `DRISHTI_Integrated_Telemedicine_System.slx`) + Track B camp-capacity model.
- `deploy/python_app/` — FastAPI scaffold (quality/demo-mode only, no ONNX yet).
- `results/` — run evidence (CSVs/logs/overlays; `*.mat` weights excluded by design).
- `reports/` — smoke CSVs, backlog traces, completion/deployment notes.

Datasets (`dataset/`) and patch caches travel separately, never committed.

## Status / honesty
Prototype public-test results, NOT clinical validation. Metrics are traced to
source files (see MODELS.md); anything untraceable is labeled "not verified".
Human-in-the-loop: an ophthalmologist makes the final call.
