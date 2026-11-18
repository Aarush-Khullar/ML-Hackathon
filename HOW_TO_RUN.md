# PathSnap – Evaluation 1 package (preprocessing + model training)

## Setup (once)
```
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Dataset location: `data_raw/train/<trip>/<trip>.track|.nodes|.arcs|.route` and `data_raw/student_test/<trip>/...` and `data_raw/evaluator_test/<trip>/...`

## Part 1 – Preprocessing & feature engineering (run in this order)
| Command | What it shows |
|---|---|
| `python src/preprocess.py 00000007` | data-quality audit of all 90 trips (missing, invalid coords, duplicates, timestamp order, gaps, speed outliers), cleaning, movement features (distance, dt, speed, bearing, turn angle). Writes `results/preprocessing/quality_report.csv` |
| `python src/preprocess_figs.py` | figure `fig_preprocessing.png` + example 32-feature table for one GPS fix |
| `python src/build_cache.py` (~45 s) | candidate road arcs + 32 model features + derived labels for all trips (creates `cache/`) |
| `python src/eda.py` (optional) | dataset inventory + EDA plots in `results/eda/` |

Explain with: `docs/PREPROCESSING_FEATURES.md`

## Part 2 – Model selection & training
| Command | What it shows |
|---|---|
| `python src/demo_model.py 00000076` (~3 s, needs only data + saved model) | loads `artifacts/lgbm_v2.joblib`, predicts road candidates for one trip, prints candidate accuracy and Top-1 (argmax and +Viterbi) |
| `python src/eval_student.py` (~15 s, no cache needed) | the saved model on all 10 student-test trips, per-trip and overall metrics |
| `python src/eval_student.py evaluator_test` (~15 s) | the same frozen model on the 10 **evaluator** trips (`data_raw/evaluator_test/`) → `results/models/eval_evaluator_test.json` |
| `python src/show_results.py` (instant) | prints the saved training results of **all 8 models** (per-fold CV scores, train vs CV gap, test metrics, confusion matrix) + summary table. `python src/show_results.py M6_LightGBM` for one model |
| `python src/run_experiments.py` (~10 min, needs `build_cache.py`) | retrains & compares all 8 models live with 5-fold trip-grouped CV + student-test, printing each fold as it finishes → `results/models/M*.json`; the full terminal output is saved to `results/models/training_log.txt` |
| `python src/train_final.py` (~20 s, needs `build_cache.py`) | trains + saves the final LightGBM → `artifacts/lgbm_v2.joblib`, `results/models/final_model_eval.json` |
| `python src/final_results.py` (~2 min, needs `build_cache.py`) | **final results**: trains all 8 models on the training trips and tests each on BOTH student_test and evaluator_test (+ the saved final model). Prints per-set tables and a combined table → `results/models/final_results_all_models.csv`, `final_results_log.txt` |
| `python src/make_figures.py` | `results/models/model_comparison_full.csv` + all figures in `results/figures/` |

Explain with: `docs/MODEL_SELECTION.md`

Saved for the final evaluation (HMM + FastAPI): `artifacts/lgbm_v2.joblib` (+ `lgbm_v2.txt` booster), `artifacts/model_config.json` (feature list, R, K, params), `artifacts/decode_config.json` (Viterbi costs), `src/decode.py`.
The old 23-feature model is kept in `artifacts/v1_baseline/` for comparison only.

## Fast demo plan (if short on time)
1. `python src/preprocess.py 00000007`
2. open `results/preprocessing/fig_preprocessing.png`
3. `python src/demo_model.py 00000076`
4. `python src/eval_student.py` and `python src/eval_student.py evaluator_test`
5. `python src/show_results.py` (training results of all 8 models)
6. open `results/models/model_comparison_full.csv` and `results/figures/fig_test_top1_f1.png`
