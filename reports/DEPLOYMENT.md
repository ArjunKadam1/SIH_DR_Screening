# Where & How to Deploy

## Option A - MATLAB (fastest for SIH internal round, 12 Sep 2026)
- **Live demo:** MATLAB Online R2026a, run `demo/runDRDemoV2.m`, `simulink/createThroughputModel.m`
- **Desktop exe:** `mcc -m runDRDemoV2 -a preprocessing -a quality -a demo` (needs MATLAB Compiler, ~500MB runtime)
- **Web:** MATLAB Web App Server - wrap `screenFundusImage` as App Designer app
- Pros: zero porting, Grad-CAM native. Cons: license/runtime heavy, no easy cloud scale.

## Option B - Python FastAPI (recommended for real deployment) - BUILT HERE
Location: `deploy/python_app/`
```
pip install -r requirements.txt
uvicorn app:app --port 8000
```
- Without ONNX: honest demo-mode (quality only). With ONNX (export via `classification/exportResNetToONNX.m`): full grading.
- **Where:** Render/Railway/HF Spaces for demo link; AWS/GCP/Azure container for hospital pilot; Docker included.
- This machine: Python 3.14 OK, Docker NOT installed - install Docker Desktop to use `docker build`.

## Option C - MATLAB Compiler SDK -> REST
`compiler.build.productionServerArchive` around `screenFundusImage` -> deploy to MATLAB Production Server on hospital LAN/PACS.

## Recommendation
SIH judging: use Option A (MATLAB, no surprises). For deployment question "where/how": answer Option B - Python+ONNX+Docker to cloud, front-end upload, GPU inference, audit log, disclaimer + referral workflow. Keep V1 .mat frozen, version ONNX, log all predictions.
