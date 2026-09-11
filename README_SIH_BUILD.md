# SIH DR Screening - Built Deliverable

## What was built now
- `demo/runDRDemoV2.m` - quality-gated demo + PDF report (fixes V1 no-gate issue)
- `demo/generateScreeningPDF.m` - one-page report with honest V1 metrics
- `simulink/createThroughputModel.m` - run in MATLAB to get DR_Throughput_Model.slx
- `classification/exportResNetToONNX.m` - MATLAB -> ONNX for deployment
- `deploy/python_app/` - FastAPI (verified: /health /metrics /screen, demo-mode on real image)
- `reports/SIH_Completion_Report.md` + `reports/DEPLOYMENT.md`

## Verified here
- Python pipeline on real Severe image: contrast 19.75 Good, demo-mode (no ONNX), acc 78.8321
- FastAPI routes OK, /health live on port 8123
- MATLAB: all 4 functions found (exist=2)

## Where & how to deploy
- SIH judging: MATLAB Online, run `runDRDemoV2`, `createThroughputModel`
- Real deploy: `deploy/python_app` -> export ONNX -> Docker/cloud (Render/Railway/HF/AWS/GCP/Azure). See README_DEPLOY.md
- This PC has no Docker; install Docker Desktop for container run.

## Further steps
Simulink timing -> 8-10 Grad-CAM saves -> pitch slides -> optional V2 (same split) -> external validation.
Limits: prototype, 78.83%, weak severe/prolif recall, Grad-CAM not diagnosis.
