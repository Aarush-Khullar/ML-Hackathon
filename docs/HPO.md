# Bayesian hyperparameter optimisation (LightGBM)

Code: `src/tune_bayes.py` · trial record: `results/tuning/optuna_trials.csv` · log: `results/tuning/tuning_log.txt` · best: `results/tuning/best_params.json`

1. **Hyperparameters searched** and 2. **ranges** (TPE samples log-uniformly where marked):

| Hyperparameter | Range | Scale | Best value |
|---|---|---|---|
| `n_estimators` | 150 – 900 | linear | 758 |
| `learning_rate` | 0.01 – 0.2 | log | 0.01238 |
| `num_leaves` | 15 – 255 | log | 122 |
| `min_child_samples` | 10 – 200 | log | 74 |
| `subsample` | 0.5 – 1.0 | linear | 0.8788 |
| `colsample_bytree` | 0.4 – 1.0 | linear | 0.9394 |
| `reg_lambda` | 0.001 – 10.0 | log | 0.288 |
| `reg_alpha` | 0.001 – 10.0 | log | 0.8512 |

Why these ranges: number of trees × learning rate trades bias against variance; `num_leaves` and `min_child_samples`
control tree complexity (the untuned model already showed a ~0.03 train–CV gap); `subsample`/`colsample_bytree` add
randomness; L1/L2 penalties regularise leaf values.

3. **Objective**: mean macro-F1 over 5-fold GroupKFold (grouped by trip) on the 80 training trips, the same metric as the
   model comparison. Student-test is used only for the comparison below. Evaluator data is never loaded.
4. **Trials**: 40 (Optuna TPE sampler, seed 42, MedianPruner; 20 trials pruned early after 2+ folds).
   Trial 0 was the untuned default configuration. Search time 28.0 min on a 20-core laptop.
5. **Best values**: table above. Best CV macro-F1 0.9473 vs untuned 0.9466.
6. **Did tuning help?** Slightly. It gave a small gain on the selection metric and on log loss, but student Top-1 is unchanged and inference is about 4× slower:

| Model | CV macro-F1 | Student Top-1 | Student macro-F1 | Student log loss | Train time (s) | Inference (ms / 1000 fixes) |
|---|---|---|---|---|---|---|
| LightGBM untuned (M6 default) | 0.9466 | 92.34% | 95.04% | 0.0545 | 3.0 | 13.7 |
| LightGBM Bayesian-tuned | 0.9473 | 92.32% | 95.10% | 0.0534 | 10.9 | 59.7 |

Decision rule fixed in advance: choose by CV macro-F1, so the tuned model became the final model (`artifacts/lgbm_final.joblib`,
trained by `src/train_final.py`). The gain is within fold noise; the honest summary is that LightGBM was already near its
optimum on these features, and the large gains came from feature engineering and HMM decoding.
Figures: `results/tuning/fig_optuna_history.png`, `fig_optuna_importance.png`.
