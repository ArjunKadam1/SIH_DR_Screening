# MA Segmentation Campaign — Results & Summary (2026-09-21 → 2026-09-23)

Project root: `C:/Users/SHIVANYA SALES/Desktop/DR tejas/SIH_DR_Screening`
Module: `segmentation/microaneurysms/` — 61-layer U-Net, 512x512, MATLAB R2026a, RTX 4070 Ti SUPER 16GB
Data: IDRiD 81 MA images (43 train / 11 val / 27 untouched test), e-Ophtha 118+30, seed 2026
Patches: 512x512 stride 256, >=10 lesion px, neg capped 8/img → train 4948 (3915 pos / 1033 neg), val 1258 (975 pos)
Role: research/supporting-lesion-evidence, NOT clinically validated. Test-27 never touched.

---

## 1. Event timeline (step-by-step executions)

| # | Date | Branch | What ran | Outcome → next |
|---|------|--------|----------|----------------|
| 0 | 09-18 | TrackB build | `trainMAUNet_TrackB_20260918.m`, patch gen, Dice/weighted/moderated losses | 1-epoch smoke baseline |
| 1 | 09-21 12:55 | `MA_verify_20260921` | 1 epoch / 1237 iters, SGD-arm orig loss, `console_1epoch_20260921.log` | loss 1.37→1.24, EVAL50 Dice 0.004 → broken-loss suspected |
| 2 | 09-21 21:22 | `MA_ACTFIX_20260921` | Removed erroneous sigmoid-after-softmax, reran 1 epoch / 1237 iters | loss 1.14→1.02, Dice@0.5 0, gap 0.0114 → inconclusive, ordered D1/D2/D3 |
| 3 | 09-21 | `MA_ACTFIX/diag/` D1 | Eval-path check: BN/dropout count, fwd-state assignment, 6 overfit patches SGD-final weights infer vs train-mode 5-pass | gap 0.0002 → evaluation path CLEARED |
| 4 | 09-21 | `MA_ACTFIX/diag/` D2 | Loss unit test on 1 patch (357 lesion px): perfect / nothing / 0.5 / inverted / sat-wrong | ordering nothing < half → loss bug LOCALIZED, clamp exonerated |
| 5 | 09-21 | `MA_ACTFIX/overfit/` | 6-patch overfit SGD 1e-4 + Adam 1e-3, 2500 cap, Dice@best/250 | SGD 0.0086 plateau, Adam →14.79 collapse → neither passes, ref-loss ordered |
| 6 | 09-21 | `training_ACTFIX/refLossOverfit` | Wrote ref-loss script (never executed here): log-prob CE clamp 1e-6 + inv-freq weights [1,100] + batch Dice smooth=1 | design banked |
| 7 | 09-21 22:02 | `MA_STAB_20260922` | D3 ref-loss + P=Y(:,:,2,:) + clip-norm 1.0, 500/500 iters, Adam 1e-4 | all finite, dBest 0.026 sub-threshold ranking → diagnostic PASS |
| 8 | 09-21 22:44 | `MA_STAB3_20260922` | Same recipe × 3 epochs (3711 iters) | ep2 emergence dBest 0.37 → ep3 0.30 → STOP |
| 9 | 09-21 23:34 | `MA_E5_20260922` | Same × 5 epochs (6185 cap, no early stop) | ep2 0.328 → ep3-4 collapse (grad 7e19→1e30) → ep5 non-finite @5148 FAIL/STOP |
| 10 | 09-22 20:52 | `MA_LR3E5_20260922` | CHANGE ONLY LR 1e-4→3e-5, 3 epochs | ep2 0.3769 → ep3 decay 0.25, norms ≤192 → transient peak, dose-response |
| 11 | 09-22 21:05 | `MA_EVALEP2_20260922` | Inference-only: ep2 best_diag vs ep3 final, same R.sel 50 (39 pos) | ceiling + split characterized, pooled Dice 0.424 |
| 12 | 09-22 21:20 | `MA_PSPLIT_20260922` | Bg/lesion P-split, no training | world-D discrimination, smooth=1e-6 shelved |
| 13 | 09-22 21:31 | `MA_FPLOC_20260922` | FP spatial concentration check | scattered far-lesion → focal-gamma favored over boundary term |
| 14 | 09-22 21:58 | `MA_GAMMA3_20260922` | Single-var gamma 2→3, 6-patch 1000 iters SGD+Adam | same learn-collapse → gamma does NOT fix |
| 15 | 09-22/23 | `testrun_IDRiD53/54_manual` | Single-image tiled 512/256 forward, ep2 net read-only | IDRiD_53 focal dots clean disc/vessels; IDRiD_54 Dice 0.547 best single image |

---

## 2. Result tables (measured numbers first)

### T1. 1-epoch verify vs ACTFIX (1237 iters, full train + 50-patch EVAL)

| Run | Loss start→end | EVAL50 Dice | dFix | gap P(MA)-P(BG) | Verdict |
|-----|---------------|-------------|------|-----------------|---------|
| verify (sigmoid bug) | 1.3711→1.2416, mean 1.3027 | 0.0040 med 0.0019 | — | flat ~0.55 output | broken |
| ACTFIX (sigmoid removed) | 1.1427→1.0176, mean 1.0597 | — | 0.0000 | 0.01142 | inconclusive |
| ACTFIX rank | AUCpool 0.3832 APpool 0.0019 bestDice 0.0064@0.20 | G1 0 G2 0 | — | — | STOP |

### T2. D1 eval-path (6 overfit patches, final SGD-arm, 2500-iter retrain loss 1.0375 both runs)

| Mode | Dice@best mean | Per-patch | Overall |
|------|---------------|-----------|---------|
| Inference (predict) | 0.0090 | 0.0046/0.0029/0.0038/0.0294/0.0041/0.0091 | 0.0090 |
| Training-mode dropout-on 5-pass | 0.0090/0.0089/0.0087/0.0087/0.0087 | — | 0.0088 |
| Gap 0.0002 | rule: ≥0.1 artifact, ≤0.05 cleared | BN=0 Dropout=2 | **CLEARED** |

### T3. D2 loss unit test (lesion 357 px; Dice per-batch smooth=1, focal eps=1e-6 γ=2 no alpha)

| Prediction | Loss (Dice+CE) | Expected order |
|------------|---------------|----------------|
| perfect (P=GT) | 0.0000 | lowest ✓ |
| predict-nothing (P=0) | 1.0160 (0.9972+0.0188) | should be > 0.5 ✗ |
| constant 0.5 | 1.1706 (0.9973+0.1733) | should be < nothing ✗ |
| inverted / sat-wrong | 14.8155 finite | huge but finite → clamp OK |
| **Reading** | nothing ≺ half → loss prefers all-zero; perfect visible (0.0) yet gradients walk to zero | **loss bug localized** |

### T4. 6-patch overfit (2500 cap, Dice@best/250)

| Arm | Final loss | Final Dice@best | Trajectory |
|-----|-----------|-----------------|------------|
| SGD 1e-4 | 1.0375 | 0.0086 plateau | flat 0.004-0.008, no pass (>0.5/500) |
| Adam 1e-3 | 14.7940 @500→2500 | 0.0027 | collapsed @500, frozen |
| Verdict | pass=0/0 | — | → D3 reference loss |

### T5. STAB 500-iter diagnostics (Adam 1e-4, clip 1.0, D3 loss)

| Iter | Loss (ce/di) | nPre→clip | dBest | gap | PMA/PBG |
|------|-------------|-----------|-------|-----|---------|
| 50 | 1.4448 | 1.49→1 | 0.0094 | -0.0082 | 0.114/0.122 |
| 250 | 1.4413 | 1.77→1 | 0.0126 | 0.0058 | 0.119/0.113 |
| 400 | 1.5821 | 5.91→1 | 0.0175 | 0.0105 | 0.106/0.096 |
| 500 | 1.5785 | 3.22→1 | 0.0260 | 0.0113 | 0.146/0.135 |
| Summary | clipRate 0.808, norms 0.75-14.82, no explosion | dFix 0.0 (P<0.5) | 6.5× floor | weak rising ranking |

### T6. Multi-epoch core (same seed/order/init; trainMean / valLoss / dFix / dBest / gap / clip / maxNorm)

| Branch | EP1 | EP2 | EP3 | EP4 | EP5 |
|--------|-----|-----|-----|-----|-----|
| STAB3 (LR 1e-4, 3711) | 1.47/0.993/0.0003/0.037/0.046/0.82/353 | 1.268/0.797/**0.312/0.371**/0.550/0.96/64 | 1.057/0.807/0.246/0.305/0.628/0.99/106 | — | — |
| E5 (LR 1e-4, 6185) | 1.446/0.978/0.037/0.078/0.348/0.87/68 | 1.167/0.774/**0.256/0.328**/0.558/0.99/152 | 1.356/0.771/**0/0**/0/1.0/7e19 | 3.507/0.771/0/0/0/1.0/1e30 | non-finite@5148 FAIL |
| LR3E-5 (LR 3e-5, 3711) | 1.427/0.978/0.051/0.071/0.225/0.96/72 | 1.222/0.786/**0.317/0.377**/0.467/0.99/192 | 1.061/0.872/0.174/0.252/0.710/1.0/170 | — | — |
| Best | — | best_diag ep2 0.377, best_val ep2 0.786 | — | — | — |

Dose-response: LR bounds violence (≤192 vs 1e19-1e30) but decay 0.317→0.174 persists → return to D2 all-zero attractor, not pure optimizer bottleneck.

### T7. EVALEP2 ceiling (ep2 net, R.sel 50 = 39 pos) + ep2→ep3 split

| Metric | Value |
|--------|-------|
| dFix mean/med, frac>0.1 | 0.3704 / 0.4070, 31/39 |
| dBest mean | 0.4480 |
| Lesion hits / patches-hit | 69/152 = 0.454, 31/39 |
| Neg FPfrac | 0.00034 |
| ΔdFix mean/med, nNeg, maxNeg | -0.1506 / -0.1732, 29/39, -0.6139 |
| ΔdBest mean/med | -0.1301 / -0.1298 |
| gap2 → gap3 | 0.488 → 0.733 (rising gap, falling overlap) |
| FULLVAL pooled Dice@0.5 | 0.4244 (inter 254621 pred 682286 gt 517717) |
| Maps | hit_hi/lo/med + fp_hi (2 red dots: vessel crossing + plain bg) |

### T8. PSPLIT (P distributions, 26332 les px vs 1308109 bg sub)

| Net | les p50/p90/p99 | bg p50/p90/p99 | @0.5 les/bg frac | per-patch les-p90 med, ≥0.5 |
|-----|----------------|---------------|------------------|---------------------------|
| ep2 | 0.3995/0.9980/1.0 | 0.0001/0.0018/0.0262 | 0.4795/0.00157 | 0.9789, 31/39 |
| ep3 | 0.9956/1.0/1.0 | 0.0018/0.0173/0.3886 | 0.7086/0.00872 | 0.9998, 34/39 |
| Rule | les-p90<0.3 & bg-p99<0.1 → magnitude → smooth 1e-6 else localization | world-D discrimination | smooth shelved | — |

### T9. FPLOC (FP mass location)

| Net | FP px | medDist | ≤10 px | >50 px | patches w/ FP |
|-----|-------|---------|--------|--------|---------------|
| ep2 | 19717 | 39.3 | 0.394 | 0.436 | 33/39 |
| ep3 | 104686 | 95.3 | 0.243 | 0.655 | 39/39 |
| Reading | 0.16%→0.87% bg>0.5, scattered far-lesion not rim → focal-gamma (general) > boundary term; threshold-only fix viable if ep2 separation suffices |

### T10. GAMMA3 (sole change γ 2→3, 6-patch 1000 iters)

| Arm | @250 | @500 | @750 | @1000 | Loss trajectory |
|-----|------|------|------|-------|-----------------|
| SGD | 0.0034 | 0.0048 | 0.0043 | 0.0050 | 1.08→1.02 flat |
| Adam | 0.0368 | 0.0000 (loss 3.95) | 0.0095 | 0.0000 | learn-collapse, same as γ=2 |
| Verdict | gamma does NOT fix | — | — | — | STOP |

### T11. Single-image test runs (ep2 net, tiled 512/256, read-only)

| Image | Size | GT MA px | thr 0.5 Dice (pred px) | best thr Dice | Artifacts |
|-------|------|----------|------------------------|---------------|-----------|
| IDRiD_53 | — | — | focal confident dots, clean disc/vessels (qualitative) | — | `testrun_IDRiD53_manual/overlay_pred05.png, fundus_GTdots.png, P_heatmap.png` |
| IDRiD_54 | 2848x4288 | 24011 | **0.5469** (pred 25489) | 0.5496 @0.40 | `testrun_IDRiD54_manual/` same 3 files, 176 tiles ~0.1 min |

IDRiD_54 is the strongest single-image MA score: pred count matches GT, above pooled 0.424.

---

## 3. Standing interpretation (no new claims)

1. Evaluation path cleared (T2); failure inside loss preference ordering (T3) with clamp exonerated.
2. D3 ref-loss can express lesions (ep2 0.33/0.37/0.38 across STAB3/E5/LR3E-5 neighborhood) but optimization is transient peak-then-decay; LR controls violence not return to attractor (T6).
3. ep2 is real reproduced peak (same metric, independently timestamped) but uncharacterized broad-vs-handful until EVAL (T7): broad (31/39 contribute) with under-segmented rims (cores hit, rims missed).
4. ep3 split is uniform slide (29/39 negative-small) + rising gap with falling overlap; FP mass scattered (T8-T9) → formulation worth one more stabilization attempt (smooth=1e-6) vs pivot/write-up decision per locked rule.
5. Gamma-3 exonerated (T10). For SIH supporting-evidence role, ep2 (`best_diagnostic_checkpoint.mat`, pooled 0.424, IDRiD_54 0.547) is sufficient to wire into `runDRDemoV2` with honest `0.42 val, test untouched` label; stop tuning unless patch-pivot ordered.

## 4. Banked artifacts

- Best: `results/MA_LR3E5_20260922/best_diagnostic_checkpoint.mat` (dBest 0.3769) + `best_val_checkpoint.mat`
- Full-val: `results/MA_EVALEP2_20260922/perpatch_EVALEP2.csv, evalEP2.mat, console_eval.log`, 4 maps
- Diagnostics: `MA_ACTFIX/diag/diagD1_finalSGDnet.mat`, `MA_PSPLIT/psplit_ep*.mat`, `MA_FPLOC/fpdist_ep*.mat`, `MA_GAMMA3/g3_*.csv`
- Logs: `console_*.log` + `train_*.csv` per branch (see T5-T6)
- Manual: `testrun_IDRiD53_manual/` + `testrun_IDRiD54_manual/` overlays/heatmaps
- Scripts (new files only, frozen preserved): `training_*/trainMA_*.m`, `manualTestRun_IDRiD5*.m`, `training_EVALEP2/PSPLIT/FPLOC/GAMMA3/*.m`

Test-27 IDRiD untouched throughout. No augmentation, no arch change, rng(42) fixed.
