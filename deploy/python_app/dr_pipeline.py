"""DR screening pipeline - Python parity of MATLAB prototype.
Mirrors: qualityCheck.m (threshold 11.45), qualityCheckV2.m,
preprocessImage.m (Lab-L CLAHE), screenFundusImage.m decision logic.

Model V1 honest metrics (DO NOT CHANGE - prototype results):
  5-class test accuracy 78.8321%
  Referable sensitivity 84.75%, specificity 97.54%, F1 90.00%
"""
from __future__ import annotations
import io
import os
import numpy as np
from PIL import Image, ImageOps

CLASS_NAMES = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]
CLASS_TO_GRADE = {"No_DR": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Proliferate_DR": 4}
REFERABLE_CLASSES = {"Moderate", "Severe", "Proliferate_DR"}

# Honest prototype metrics from MATLAB Model V1
MODEL_V1_METRICS = {
    "model": "ResNet-18 (MATLAB trained_resnet18.mat)",
    "test_accuracy": 78.8321,
    "train_accuracy_final": 96.8750,
    "val_accuracy_final": 76.5455,
    "best_val_accuracy": 79.4545,
    "overfitting_note": "Train 96.88% vs Val 76.55% - strong evidence of overfitting. Preserve V1 as baseline.",
    "referable": {"sensitivity": 84.75, "specificity": 97.54, "precision": 95.94, "f1": 90.00},
    "disclaimer": "Prototype public-test results. NOT clinical validation.",
}

ONNX_PATH = os.environ.get("DR_ONNX_PATH", "model/resnet18_dr.onnx")


def load_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(img)


def quality_check_v1(img_rgb: np.ndarray):
    """Port of qualityCheck.m: std(gray) >= 11.45 -> Good."""
    # ITU-R BT.601 luma like rgb2gray
    gray = 0.2989 * img_rgb[:, :, 0] + 0.5870 * img_rgb[:, :, 1] + 0.1140 * img_rgb[:, :, 2]
    contrast = float(np.std(gray))
    flag = "Good" if contrast >= 11.45 else "Low Contrast"
    return contrast, flag


def quality_check_v2(img_rgb: np.ndarray):
    """Port of qualityCheckV2.m thresholds."""
    gray = (0.2989 * img_rgb[:, :, 0] + 0.5870 * img_rgb[:, :, 1] + 0.1140 * img_rgb[:, :, 2]) / 255.0
    contrast = float(np.std(gray))
    brightness = float(np.mean(gray))
    # Laplacian variance blur approx with numpy
    lap = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    blur = float(np.var(lap))
    reasons = []
    if contrast < 0.045:
        reasons.append("Low contrast")
    if brightness < 0.08:
        reasons.append("Too dark")
    elif brightness > 0.75:
        reasons.append("Too bright")
    if blur < 0.0005:
        reasons.append("Possible blur")
    flag = "Good" if not reasons else "Needs Review"
    return flag, ", ".join(reasons) if reasons else "Image passed all quality checks", {
        "Contrast": contrast, "Brightness": brightness, "BlurScore": blur}


def enhance_image(img_rgb: np.ndarray) -> np.ndarray:
    """Lightweight parity of preprocessImage.m Lab-L CLAHE.
    Full CLAHE needs OpenCV; here we do autocontrast on L channel as
    deploy-friendly approximation. MATLAB remains reference for SIH demo.
    """
    try:
        import cv2  # optional if present
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=0.005 * 100, tileGridSize=(8, 8))
        # cv2 ClipLimit is in different scale; use 2.0 as safe approx
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        out = cv2.merge([l2, a, b])
        return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
    except Exception:
        # Fallback: PIL autocontrast per channel (documented approximation)
        pil = Image.fromarray(img_rgb)
        return np.array(ImageOps.autocontrast(pil, cutoff=0.5))


def try_onnx_predict(img_rgb: np.ndarray):
    """If DR_ONNX_PATH exists, run onnxruntime. Else return None (demo mode)."""
    if not os.path.exists(ONNX_PATH):
        return None
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        inp = Image.fromarray(img_rgb).resize((224, 224))
        arr = np.array(inp).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[None, :]
        out = sess.run(None, {sess.get_inputs()[0].name: arr})[0][0]
        # softmax
        e = np.exp(out - out.max())
        probs = e / e.sum()
        idx = int(np.argmax(probs))
        return CLASS_NAMES[idx], float(probs[idx]), [float(p) for p in probs]
    except Exception as e:
        return {"error": str(e)}


def screen_image(file_bytes: bytes) -> dict:
    img = load_image(file_bytes)
    contrast, qflag = quality_check_v1(img)
    q2flag, q2reason, q2m = quality_check_v2(img)
    enhanced = enhance_image(img)

    onnx = try_onnx_predict(np.array(Image.fromarray(enhanced).resize((224, 224))))
    if onnx is None:
        # Demo mode: DO NOT fabricate a diagnosis. Return quality+enhancement
        # and honest V1 metrics, require MATLAB/ONNX for real grade.
        return {
            "pipelineVersion": "Python-Deploy-v1",
            "quality_v1": {"contrast": contrast, "flag": qflag},
            "quality_v2": {"flag": q2flag, "reason": q2reason, "metrics": q2m},
            "classificationPerformed": False,
            "predictedClass": None,
            "ICDRGrade": None,
            "referableDR": None,
            "mode": "DEMO-MODE-NO-ONNX",
            "message": "No ONNX model found at model/resnet18_dr.onnx. Export MATLAB net via classification/exportResNetToONNX.m then set DR_ONNX_PATH.",
            "modelV1Metrics": MODEL_V1_METRICS,
            "disclaimer": "AI-assisted screening prototype. Clinical confirmation required. Grad-CAM is attention, not lesion proof.",
        }
    if isinstance(onnx, dict) and "error" in onnx:
        return {"error": onnx["error"], "modelV1Metrics": MODEL_V1_METRICS}
    pred, conf, probs = onnx
    grade = CLASS_TO_GRADE[pred]
    referable = grade >= 2
    return {
        "pipelineVersion": "Python-Deploy-v1",
        "quality_v1": {"contrast": contrast, "flag": qflag},
        "quality_v2": {"flag": q2flag, "reason": q2reason, "metrics": q2m},
        "classificationPerformed": True,
        "predictedClass": pred,
        "confidence": conf,
        "allScores": dict(zip(CLASS_NAMES, probs)),
        "ICDRGrade": grade,
        "referableDR": referable,
        "referralRecommendation": "REFER FOR OPHTHALMOLOGIST REVIEW" if referable else "NO IMMEDIATE REFERRAL - ROUTINE SCREENING",
        "modelV1Metrics": MODEL_V1_METRICS,
        "disclaimer": "AI-assisted screening prototype. Clinical confirmation required.",
    }
