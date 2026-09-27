# Model weights (download, not cloned)

The trained nets are too large for git (`results/**/*.mat`, `*.onnx` ignored).
Download the 4 files from the GitHub Release **`v0.1-prototype`** and place
them at exactly these paths (filenames + variable names matter):

| File (place at this path) | Size | Variable | Source checkpoint | Val metrics (source) |
|---|---|---|---|---|
| `HandheldDR/results/handheld_resnet50_aptosidrid4.mat` (sibling folder `HandheldDR/`, NOT in this repo) | ~90 MB | `trainedNetHandheld` | R50 4-class, APTOS + IDRiD-train | APTOS-test n=548: acc 82.30%, referable sens 91.93% / spec 95.08% / F1 92.34 — `HandheldDR/reports/all_models_report.txt:51-65` (13-Sep-2026) |
| `results/MA_LR3E5_20260922/best_diagnostic_checkpoint.mat` | ~115 MB | `bestNet` | MA U-Net, ep2, LR 3e-5 | pooled val Dice 0.424, dBest 0.3769 — `results/MA_SUMMARY_20260923.md` T6–T7, `summary_LR3E5.mat` |
| `results/VES_P3_NORM_20260923/arm_resize/armA_block2_iter80.mat` | ~1.8 MB | `net` | P3-A vessel, block 2 | DRIVE-val 37–40 pooled dFix 0.3735, thinRecall ~0.73 — `eval_P3_ArmA.csv` |
| `results/temperature_handheld.mat` | tiny | `T` | T=2.5 handheld calibration (25-Sep-2026) | NLL 0.4759, ECE 0.109→0.042 — `results/fit_temp_handheld.log` |
| `results/handheld_r50_aptosidrid4.onnx` (copy to `deploy/python_app/model/`) | ~94 MB | — | ONNX export of the R50 above (`classification/exportResNetToONNX.m`; needs ONNX Converter add-on) | parity vs MATLAB `classify` exact to 4 decimals on identical pixels |

Temperature file ships in-repo (tiny). The R50 net lives in the sibling
`HandheldDR/` project; the Release bundles a copy for fresh clones.
Local Drop-in: you may instead place `handheld_resnet50_aptosidrid4.mat` in an
ignored top-level `models/` folder — `exportResNetToONNX.m` checks there first,
then falls back to a file picker (no absolute paths).

## Run (MATLAB R2026a + Deep Learning + Image Processing toolboxes)
```
addpath('demo'); app = RetinaAIApp_Lesion;   % new app (old RetinaAIApp kept as fallback)
```
Upload a **raw** fundus photo → ANALYZE → tick lesion checkbox → upload FOV
mask (vessel only) → ANALYZE → Lesion tab → Report tab → Generate PDF.

Research prototype — NOT clinically validated. Test-27 IDRiD untouched.
