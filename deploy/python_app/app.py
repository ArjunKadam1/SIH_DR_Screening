import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from dr_pipeline import screen_image, MODEL_V1_METRICS, LIVE_R50_METRICS, CLASS_NAMES, ONNX_PATH

app = FastAPI(title="SIH DR Screening - Explainable AI Prototype", version="1.0.0")


def _live_status():
    found = os.path.exists(ONNX_PATH)
    return {
        "status": "ok",
        "live_model": "handheld R50 4-class ONNX (Mild/Moderate/No_DR/SevereProlif)",
        "onnx_present": found,
        "grading": "LIVE" if found else "DEMO-MODE-NO-ONNX",
        "traced": LIVE_R50_METRICS,
        "historical_baseline_v1": {
            "note": "Superseded V1 baseline for reference only - NOT the live model.",
            **MODEL_V1_METRICS,
        },
    }


@app.get("/health")
def health():
    return _live_status()


@app.get("/metrics")
def metrics():
    return _live_status()


@app.get("/", response_class=HTMLResponse)
def home():
    live = os.path.exists(ONNX_PATH)
    state = ("Grading: <b>LIVE</b> (ONNX model found)."
             if live else
             "Grading: <b>demo-mode</b> (no ONNX found - quality checks only).")
    return """
    <html><head><title>SIH DR Screening</title></head><body style="font-family:sans-serif;max-width:720px;margin:40px auto">
    <h2>Explainable AI for Diabetic Retinopathy Screening - SIH 2026 Prototype</h2>
    <p><b>Live model:</b> handheld R50 4-class ONNX (Mild / Moderate / No_DR / SevereProlif).
    Traced: APTOS-test n=548, acc 82.30%%, referable sens 91.93%% / spec 95.08%%.</p>
    <p>%s</p>
    <p><b>Pipeline:</b> upload fundus image -&gt; quality check -&gt; enhancement -&gt; grade -&gt; referable decision</p>
    <p style="color:#a00">Prototype only. Clinical confirmation required. Grad-CAM is attention, not lesion proof.</p>
    <form action="/screen" enctype="multipart/form-data" method="post">
    <input type="file" name="file" accept="image/*"><input type="submit" value="Screen">
    </form>
    <p>API: POST /screen with image file. GET /health, /metrics</p>
    <h4>Historical baseline (V1, superseded - not the live model)</h4>
    <p>V1 ResNet-18: test accuracy 78.83%%, referable sens 84.75%% / spec 97.54%%. Preserved for reference.</p>
    <p>To enable real grading: install the ONNX Converter add-on in MATLAB, run MATLAB classification/exportResNetToONNX.m, place model at model/handheld_r50_aptosidrid4.onnx</p>
    </body></html>
    """ % state


@app.post("/screen")
async def screen(file: UploadFile = File(...)):
    data = await file.read()
    try:
        result = screen_image(data)
        # numpy types -> native
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
