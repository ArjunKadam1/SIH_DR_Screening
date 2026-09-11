# SIH 2026 Project Completion Report - Explainable AI for DR Screening (ID 26038)

## 1. Summary
End-to-end MATLAB prototype: fundus image -> quality check -> Lab-L CLAHE enhancement -> ResNet-18 5-class -> ICDR grade -> referable decision -> Grad-CAM overlay -> text/PDF report. Dataset: APTOS-derived 224x224 Gaussian-filtered, 3662 images (No_DR 1805, Mild 370, Moderate 999, Severe 193, Proliferate 295). Split 70/15/15 (train 2564 / val 550 / test 548). Demo `demo/runDRDemo.m` + V2 `screenFundusImage.m` working.

## 2. Real results (DO NOT inflate)
- Test accuracy: **78.8321%**
- Per-class F1: No_DR 97.08%, Mild 65.63%, Moderate 69.20%, Proliferate 32.79%, Severe 40.00%
- Referable (grade>=2): Sens **84.75%**, Spec **97.54%**, Prec 95.94%, F1 90.00% (TP 189, TN 317, FP 8, FN 34)
- Overfitting: train 96.875% vs val 76.545%, best val 79.45% @ iter 740. Keep V1 as baseline.
- Quality: V1 contrast-std threshold 11.45 (prototype, n=50, 10th pct). V2 adds blur/illumination/FOV gate.
- Grad-CAM: `gradCAM(net,img,pred,'ReductionLayer','pool5')` - attention only, NOT lesion proof.

## 3. What was built in this completion pass
- `demo/runDRDemoV2.m` - quality-gated demo + PDF report
- `demo/generateScreeningPDF.m` - one-page report helper
- `simulink/createThroughputModel.m` - throughput/bottleneck model (arrival -> AI time -> review capacity -> queue)
- `classification/exportResNetToONNX.m` - deploy path MATLAB -> ONNX
- `deploy/python_app/` - FastAPI + Dockerfile, runs without MATLAB, demo-mode until ONNX dropped in
- `reports/DEPLOYMENT.md` - where & how to deploy

## 4. Further steps (priority order)
1. Simulink: run creator, set measured inference time, show break-even at 100k pts/yr
2. Save 8-10 good Grad-CAM examples (correct + sensible heatmaps)
3. PDF report polish + pitch slides (problem, arch, demo, honest metrics, bottleneck, limits, roadmap)
4. Optional V2 model: same split, stronger aug/dropout, lower LR, early stop - compare honestly
5. Segmentation: keep to optic-disc/vessel demo only; rest = future work
6. External validation: IDRiD/Messidor-2, multi-site; clinical validation out of scope for internal round (12 Sep 2026)

## 5. Limitations (state clearly in pitch)
Prototype not clinically validated; 78.83% below clinical target; weak severe/proliferative recall; small filtered dataset; quality threshold prototype; no hard-stop in V1 demo (fixed in V2); Grad-CAM not diagnosis; no hardware/field deployment; no full lesion segmentation.

## 6. How to demo
MATLAB: `runDRDemoV2` -> select image -> 2x2 figure + `reports/screening_report_<name>.pdf`. Python: `uvicorn app:app` -> upload at localhost:8000.

## 7. V2 retrain results 2026-09-09 (honest, same V1-exact split 2564/550/548, GPU)
- Config A (frozen conv1-res4, LR 1e-4, aug + balanced 6320 train, patience 8): stopped @ ~2 epochs (iter 300, val criterion). Test **69.16%**, refer sens **47.98%** / spec 99.38% / F1 64.46%. Per-class F1: Mild 47.9, Moderate 40.0, No_DR 93.7, Prolif 41.7, Severe 32.0. Files: `results/trained_resnet18_v2_A.mat`, `v2_training_metrics_A.mat`.
- Config B (full fine-tune, LR 1e-5, same data): stopped @ ~5 epochs (iter 900). Test **72.99%**, refer sens **56.05%** / spec 99.08% / F1 71.23%. Per-class F1: Mild 52.4, Moderate 59.2, No_DR 92.9, Prolif 21.8, Severe 34.8. Files: `results/trained_resnet18_v2_B.mat`, `v2_training_metrics_B.mat`.
- Verdict: both V2 configs UNDERPERFORM V1 (78.83% / sens 84.75%) with current hyperparams. B > A, so full fine-tune beats frozen head here, but neither clears the bar. Early stopping fired quickly (val peaked ~66% then fell) = underfit, not overfit. Checkpoint best-selection did not engage (MATLAB DAG checkpoints lack BN stats for direct classify; fell back to last-epoch net).
- Cache: `results/v2_cache_train/` (6320) + `v2_cache_val/` (550) 224px PNGs built once (~10 min), reused. V1 file untouched.
- Next experiments (not run): patience 15-20, LR sweep (A: 3e-4, B: 3e-5/1e-4), lighter augmentation, no-oversample ablation, train full 30 epochs without early stop and pick by val curve. Keep V1 as the pitched baseline until a V2 honestly beats it.
- FULL-30 follow-up 2026-09-09 (same configs, `ValidationPatience` 10000 = no early stop, all else identical):
  - V2-A-full30: test **78.28%**, sens **82.51%** / spec 95.08% / F1 87.00%. Per-class F1: Mild 56.9, Moderate 72.4 (beats V1's 69.2), No_DR 95.4, Prolif 25.5, Severe 19.0. Gap to V1 closed from -9.7pp to -0.55pp: early stopping was the culprit behind the first A run.
  - V2-B-full30: test **77.55%**, sens **69.96%** / spec 98.46% / F1 81.25%. Per-class F1: Mild 58.5, Moderate 70.2, No_DR 94.9, Prolif 27.6, Severe 29.3. Better than B-earlystop (72.99%) but sens still trails V1 by ~15pp.
  - Verdict after 4 V2 attempts: V1 (78.83% / 84.75%) remains champion. V2-A-full30 is the validated runner-up and the model to iterate from. Both full-30 nets saved over the early-stop files (same names, full-epoch weights).
- Live 5-sample unseen check 2026-09-09 (held-out test images, never trained on; same `runDRDemoV2` pipeline): V1 exact-grade 3/5 (No_DR, Mild, Moderate correct; Severe/Prolif graded Moderate but correctly REFERRED => referral 5/5). V2-A exact-grade 1/5 (only Mild; No_DR false-alarmed Moderate @55%, Moderate missed as Mild => referral 4/5; Severe/Prolif graded Moderate but correctly referred). Matches full-test metrics: V1 keeps the demo.
- Tier-1 scale-robustness experiment 2026-09-10 (same V1-exact 548 test set, demo enhancement path so absolutes differ slightly from published training-path numbers; files: `preprocessing/cropToFOV.m`, `classification/ttaClassify.m`, opt-in `'Crop'`/`'TTA'` flags in `runDRDemoV2`/`screenFundusImage`, defaults OFF so current behavior is byte-identical — verified Moderate 71.9443 reproduced exactly):
  - FOV-crop: engaged 0/548 (test images are pre-cropped, no black borders; gate failed closed as designed). No effect on this data — code kept dormant for foreign camera images, demo default unchanged.
  - TTA-5 (V1): acc 76.09→72.08, sens 87.00→91.48, spec 94.77→84.00. TTA shifts the operating point toward REFER: +4.5pp sensitivity for -4pp accuracy / -11pp specificity.
  - TTA-5 (V2-A): acc 70.26→59.49, sens 82.51→93.27 (first config ever to clear the 90% clinical bar), spec 86.15→60.00. Same shift, more extreme — unshippable as-is (40% false alarms).
  - Decision: demo defaults UNCHANGED (base-single). Tier 1 = documented negative on accuracy, useful finding on the sensitivity lever. Candidates: TTA-as-second-opinion for low-confidence singles, TTA threshold tuning, or Tier-2 retrain. Full numbers: `results/tier1_rescore.mat`.
- Second-opinion safety net 2026-09-10 (SHIPPED, default ON in `runDRDemoV2`; `screenFundusImage` default stays OFF): single view non-referable with conf<70 auto-consults TTA-5; escalation is referral-only, never downgrades. Measured on 548 (V1): sens 87.00→**90.13** (clears the 90% bar), spec 94.77→92.31, only 38 consults (6.9%, negligible cost), 15 escalations (7 correct / 8 false alarms). V2-A: sens 82.51→87.44, spec 86.15→81.23. Numbers: `results/so_rescore.mat`.
- Pipeline unification (same change): demo base path now single-enhances (passes full-res input; `screenFundusImage` enhances once) instead of double-CLAHE. Proven identical on all 548 test-224 images; on full-res camera images double-enhance was flipping verdicts (observed: a true-Moderate 1424px image read Mild 56% double vs Moderate 95% single). 224 behavior verified byte-identical (Moderate 71.9443 reproduced).
- Honest limits: escalation was wrong on one user-observed healthy full-res eye (Mild 50% → escalated REFER; logged as false alarm). Grade stays Mild with borderline note — the escalation text, not the grade, carries the uncertainty.
