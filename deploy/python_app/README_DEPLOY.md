# Deploy - Python FastAPI (no MATLAB needed at runtime)

## Run locally
```
cd deploy/python_app
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```
Open http://localhost:8000

## Enable real grading (recommended)
1. In MATLAB: run `classification/exportResNetToONNX.m` - exports `results/resnet18_dr.onnx`
2. Copy to `deploy/python_app/model/resnet18_dr.onnx`
3. `pip install onnxruntime` and restart. `/screen` then returns grade + referable decision.

Without ONNX, `/screen` runs in honest DEMO-MODE: quality + enhancement only, no fabricated grade.

## Docker (needs Docker Desktop - not installed on this machine)
```
docker build -t sih-dr-screening .
docker run -p 8000:8000 -v %cd%/model:/app/model sih-dr-screening
```

## Cloud options (where & how)
- **Easiest demo:** Render / Railway / HuggingFace Spaces - push this folder, set start `uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Hospital network:** MATLAB Web App Server OR `mcc -m runDRDemoV2` exe (see reports/DEPLOYMENT.md)
- **Scale:** AWS SageMaker / GCP Vertex / Azure ML with ONNX + FastAPI container
