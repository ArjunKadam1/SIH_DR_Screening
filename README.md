# DRISHTI — Explainable AI for Diabetic Retinopathy Screening (SIH 2026 prototype)

MATLAB screening workstation: upload → quality gate → R50 grading + referable
decision + Grad-CAM → optional MA / vessel lesion evidence → screening report.

## Quick start
1. MATLAB R2026a + Deep Learning Toolbox + Image Processing Toolbox (8 GB RAM, GPU optional).
2. Download the 3 model files — see **[MODELS.md](MODELS.md)** (GitHub Release `v0.1-prototype`).
3. `addpath('demo'); app = RetinaAIApp_Lesion;` (previous `RetinaAIApp` kept as fallback).
4. Upload a **raw** fundus photo → ANALYZE → Lesion tab → Report tab → Generate PDF.

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
