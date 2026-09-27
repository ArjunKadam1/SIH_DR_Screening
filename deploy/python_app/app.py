from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from dr_pipeline import screen_image, MODEL_V1_METRICS, CLASS_NAMES

app = FastAPI(title="SIH DR Screening - Explainable AI Prototype", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "model": "V1-ResNet18-prototype", "metrics": MODEL_V1_METRICS}


@app.get("/metrics")
def metrics():
    return MODEL_V1_METRICS


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html><head><title>SIH DR Screening</title></head><body style="font-family:sans-serif;max-width:720px;margin:40px auto">
    <h2>Explainable AI for Diabetic Retinopathy Screening - SIH 2026 Prototype</h2>
    <p><b>Pipeline:</b> upload fundus image -&gt; quality check -&gt; enhancement -&gt; grade (needs ONNX) -&gt; referable decision</p>
    <p><b>Model V1 honest accuracy:</b> 78.8321% | Referable Sens 84.75% Spec 97.54%</p>
    <p style="color:#a00">Prototype only. Clinical confirmation required. Grad-CAM is attention, not lesion proof.</p>
    <form action="/screen" enctype="multipart/form-data" method="post">
    <input type="file" name="file" accept="image/*"><input type="submit" value="Screen">
    </form>
    <p>API: POST /screen with image file. GET /health, /metrics</p>
    <p>To enable real grading: install the ONNX Converter add-on in MATLAB, run MATLAB classification/exportResNetToONNX.m, place model at model/handheld_r50_aptosidrid4.onnx</p>
    </body></html>
    """


@app.post("/screen")
async def screen(file: UploadFile = File(...)):
    data = await file.read()
    try:
        result = screen_image(data)
        # numpy types -> native
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
