# PathSnap – Model Selection & Training

> **Final-evaluation update.** The model comparison below (sections 1–6) is unchanged. After it, LightGBM was tuned
> with Bayesian optimisation (`docs/HPO.md`); the tuned model `artifacts/lgbm_final.joblib` replaced the hand-picked
> evaluation-1 model (now in `artifacts/archive/`). The HMM costs were re-tuned on training out-of-fold predictions
> (`docs/HMM.md`). Final numbers: student-test Top-1 92.32% → **95.18%** with HMM; evaluator-test 94.07% → **95.54%**
> (README). "+Viterbi" columns in `results/models/` use the final decoder settings.


**Run live:** `python src/demo_model.py 00000076` (one trip) · `python src/eval_student.py` (all 10 student-test trips).
Re-run everything: `python src/build_cache.py && python src/run_experiments.py && python src/train_final.py && python src/make_figures.py` (≈10 min, seed 42).

## 1. Problem formulation
For each GPS fix we generate candidate road arcs (≤12 within 60 m). The model is a **binary classifier on (fix, arc) pairs**: P(vehicle is on this arc). The highest-probability arc per fix is the prediction (Top-1). Probabilities are reused as emission scores for Viterbi decoding (light version now, full HMM in the final evaluation), so a **probabilistic** classifier is required.
Labels: derived per fix by aligning fixes to the reference `.route` (monotone DP). Candidate miss rate on student-test: 0.2 %.

## 2. Experimental protocol (no leakage)
* Train: 80 training trips (186,072 fixes with a reliable label; fitted on every 3rd fix, evaluated on all).
* **5-fold GroupKFold by trip**: no trip is split across folds. CV macro-F1 is the **selection metric**; CV Top-1, log loss and speed break ties.
* Student-test: 10 unseen trips (19,160 fixes), used only to report results. Viterbi costs were tuned on **training out-of-fold** predictions, never on student-test.
* Evaluator-test: 10 further trips (19,674 fixes) received **after** the model was frozen. They were run once through the saved model, with nothing refitted or tuned, and they share no trip id or file content with train or student-test.

## 3. Results: 8 models, 6 families (v2 features; from `results/models/model_comparison_full.csv`)
| Model | CV macro-F1 (±std) | CV Top-1 | Test cand. accuracy | Test macro-F1 | Test Top-1 | Test Top-1 +Viterbi | Test log loss | Train s |
|---|---|---|---|---|---|---|---|---|
| M1 Logistic Regression | 0.928 ±0.016 | 0.895 | 0.967 | 0.923 | 0.891 | 0.915 | 0.085 | 2.0 |
| M2 Gaussian NB | 0.856 ±0.010 | 0.878 | 0.923 | 0.847 | 0.865 | 0.897 | 0.868 | 0.1 |
| M3 Decision Tree | 0.939 ±0.019 | 0.895 | 0.972 | 0.934 | 0.894 | 0.916 | 0.102 | 27.8 |
| M4 Random Forest | 0.945 ±0.018 | 0.901 | 0.976 | 0.942 | 0.904 | **0.951** | 0.060 | 20.1 |
| M5 HistGradientBoosting | 0.946 ±0.017 | 0.902 | 0.977 | 0.945 | 0.913 | 0.950 | 0.061 | 6.1 |
| **M6 LightGBM** | **0.947** ±0.017 | **0.906** | **0.979** | **0.950** | **0.923** | 0.950 | **0.055** | 6.7 |
| M7 XGBoost | 0.946 ±0.018 | 0.905 | 0.978 | 0.947 | 0.918 | 0.948 | 0.056 | 5.6 |
| M8 MLP (64,32) | 0.939 ±0.019 | 0.900 | 0.975 | 0.939 | 0.909 | 0.942 | 0.064 | 51.3 |

**Saved final model** (`artifacts/lgbm_v2.joblib`, LightGBM 400 trees, lr 0.05, 63 leaves): candidate-level accuracy **97.9 %**, macro-F1 **94.9 %**, Top-1 **92.2 %**, Top-1 with Viterbi **94.8 %** (`results/models/final_model_eval.json`, per trip in `final_per_trip.csv`).

Figures: `results/figures/fig_cv_f1.png`, `fig_test_top1_f1.png`, `fig_logloss.png`, `fig_overfit_gap.png`, `fig_confusion_matrices.png`, `fig_mlp_loss.png`, `fig_feature_importance.png`.

## 3b. Evaluator-test results (saved model, frozen, run once)
`python src/eval_student.py evaluator_test` → `results/models/eval_evaluator_test.json` and `eval_evaluator_test_per_trip.csv`

| Metric | Student-test (19,160 fixes) | **Evaluator-test (19,674 fixes)** |
|---|---|---|
| Candidate-level accuracy | 97.89 % | **98.44 %** |
| Candidate-level macro-F1 | 94.93 % | **96.33 %** |
| Precision / recall (on-road) | 92.3 % / 89.9 % | **95.0 % / 92.2 %** |
| Log loss | 0.054 | **0.043** |
| Top-1 road accuracy | 92.23 % | **93.94 %** |
| Top-1 + Viterbi | 94.84 % | **95.58 %** |

The scores hold up on fully unseen data, so the model is not overfitted to student-test. Per trip, 7 of 10 evaluator trips are ≥ 95.5 % after Viterbi. Trip 00000030 is the weak one at 82.7 % (654 fixes). Viterbi lowers trips 17 and 59 slightly, which is the same parking/U-turn effect seen on student trip 07.

## 4. What changed from v1 → v2 (how we went from 89.5 % to 94.8 % Top-1)
| Step | Student Top-1 | Why |
|---|---|---|
| v1: 23 features, tuned LightGBM | 89.5 % | baseline |
| Error analysis | – | 4.3 % of fixes on the twin arc (wrong direction, mostly at < 2 m/s), 2.9 % on a side road sharing a node, rest at junction boundaries |
| + 9 long-range context features (heading over ±10/±20 fixes, distance/progress over ±10/±20) | 92.2 % | direction is still visible when the car crawls or stops |
| + Viterbi decoding over the road graph | **94.8 %** | consecutive fixes must stay on the same arc or move to a connected one |
| Tried and rejected: heading-aware label alignment (91.4 %), stage-2 stacking (93.4 % alone, worse under Viterbi), more data / trees (94.5–94.7 %) | – | no gain, kept simple |

## 5. Why LightGBM
* Best CV macro-F1 (0.947) and CV Top-1 (0.906). Also best on student-test for candidate accuracy, macro-F1, Top-1 and log loss.
* Honest caveat: the fold std (≈0.017) is larger than the gaps between the four tree ensembles, so "best" is a modest claim. After Viterbi, RF/HistGB/LightGBM are all ≈95 %.
* Tree boosting captures the non-linear interactions that matter (distance **gap** to the nearest candidate × heading agreement × context). Linear models lose ~2 F1 points. Naive Bayes fails because the features are strongly correlated (log loss 0.87, badly calibrated).
* Best calibrated probabilities (log loss 0.055), which makes it the best HMM emission model. Trains in 7 s, inference ≈ 0.06 s per 1,500-fix trip.
* Over/under-fitting: train macro-F1 is close to CV (see `fig_overfit_gap.png`), so there is no serious overfitting.
* Top features (gain share): `dist_gap` 43 %, `cos_hd_w20` 14 %, `ctx_prog_w20` 12 %, `dist_rank` 7 %. Two of the new long-range features are in the top 3.

## 6. Which "accuracy"? (be ready for this question)
* **Candidate-level accuracy 97.9 %**: correct yes/no for each (fix, arc) pair. This is inflated by the 88 % negatives.
* **Macro-F1 94.9 %**: the balanced version of that number. This is the selection metric.
* **Top-1 road accuracy 92.2 % → 94.8 % with Viterbi**: the right arc chosen for each fix. This is the strictest and most meaningful number.
* Remaining errors: parked or crawling vehicles next to two-way roads (trip 07 parks at the end, so Viterbi drops it to 86 %), and fix-to-arc boundaries at junctions where the derived label itself is ambiguous within GPS noise.

## 7. Saved for the final evaluation
`artifacts/archive/lgbm_v2.joblib` (evaluation-1 model; final model is `artifacts/lgbm_final.joblib`), `artifacts/model_config.json` (32 features, R = 60 m, K = 12, params), `artifacts/decode_config.json` + `src/decode.py` (Viterbi). The v1 model is kept in `artifacts/v1_baseline/` for comparison.

## 8. Likely viva questions
* *Why classify (fix, arc) pairs instead of predicting a road id directly?* Road ids are arbitrary per-trip integers in different cities, so they cannot be learned as classes. Ranking candidates generalises across cities.
* *Why macro-F1?* The data is imbalanced (≈12 % positives). Accuracy would be 88 % just by predicting "no".
* *Why GroupKFold?* Fixes within a trip are highly correlated, so random splits leak and inflate scores.
* *Did you tune on the test set?* No. Model choice used grouped CV, and the Viterbi costs used training OOF predictions. Student-test is only reported.
* *Why train on every 3rd fix?* 1 Hz fixes are near-duplicates. Training on all fixes gave no gain (94.7 % vs 94.8 %).
* *Is Viterbi part of eval 1?* It is a light first version for evaluating candidate-road prediction quality. The full HMM (gap-aware transitions, route reconstruction) and the FastAPI service are for the final evaluation.
