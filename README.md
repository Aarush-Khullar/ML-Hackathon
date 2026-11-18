# ML-Hackathon
ML hackathon of team 5

TEAM NUMBER - 5

# TEAM MEMBERS LIST:

1) PES2UG24CS014 - Aarush Khullar

2) PES2UG24CS044 - Aks Raj Singh

3) PES2UG24CS035 - Aditya Sehal

4) PES2UG24CS046 - Akshat Kumar

---

# PathSnap – Distributed Map-Matching & Route Reconstruction

PathSnap matches noisy 1 Hz GPS trajectories to a road network. For every GPS fix it generates candidate road
arcs, scores them with a LightGBM classifier (32 engineered features), and reconstructs a coherent route with an
HMM/Viterbi decoder over the road graph. A FastAPI server (laptop 1) runs the pipeline. A browser UI (laptop 2) lets the
evaluator pick one of the 10 reserved trips and shows the predicted route and the accuracy computed on the server.

```
 .track/.nodes/.arcs ─► preprocess ─► candidates (≤12 arcs ≤60 m) ─► 32 features ─► LightGBM P(arc|fix)
                                                                                         │
                     UI (laptop 2) ◄── JSON route + metrics ◄── FastAPI (laptop 1) ◄── HMM / Viterbi
```

## Results

Final model: Bayesian-tuned LightGBM (`artifacts/lgbm_final.joblib`) + HMM/Viterbi (`artifacts/decode_config.json`).

| Held-out set | Candidate accuracy | Macro-F1 | Log loss | Top-1 (classifier) | Top-1 (+HMM) |
|---|---|---|---|---|---|
| Student-test (10 trips, 19,160 fixes) | 97.96% | 95.10% | 0.0534 | 92.32% | **95.18%** |
| Evaluator-test (10 trips, 19,674 fixes) | 98.45% | 96.36% | 0.0428 | 94.07% | **95.54%** |

Route-level effect of the HMM on student-test: route breaks 646 → 7, continuity 96.6% → 100.0%,
route F1 90.8% → 94.7%.
Top-1 = share of GPS fixes whose predicted arc equals the derived reference arc (main metric). Candidate accuracy is inflated
by ~88 % easy negatives and is reported only for completeness. 8-model comparison: `results/models/model_comparison_full.csv`;
all models on both sets: `results/models/final_results_all_models.csv`. Evaluator numbers were computed after every
choice was frozen and were not used for any decision.
Method notes: `docs/PREPROCESSING_FEATURES.md`, `docs/MODEL_SELECTION.md`, `docs/HPO.md`, `docs/HMM.md`.

## Repository layout
| Folder | Contents |
|---|---|
| `src/` | all pipeline code: loading/projection (`data_utils.py`), audit & cleaning (`preprocess.py`), EDA (`eda.py`), candidates/labels/features (`features.py`, `build_cache.py`), 8-model comparison (`models_zoo.py`, `run_experiments.py`, `train_all.py`, `show_results.py`, `make_figures.py`), Bayesian optimisation (`tune_bayes.py`), final model (`train_final.py`), HMM/Viterbi (`hmm.py`, `hmm_experiments.py`), inference pipeline (`pipeline.py`), Folium maps (`folium_maps.py`, `screenshot_maps.py`), report (`make_report.py`) |
| `api/` | FastAPI backend (`main.py`): endpoints, model loading, server-side scoring, request logs |
| `ui/` | evaluator UI (`index.html` with Leaflet, `serve_ui.py` static server) |
| `artifacts/` | saved models and configs: final classifier, `model_config.json` (features, radius, K, params), `decode_config.json` (HMM parameters) |
| `results/` | experiment outputs: `eda/`, `preprocessing/`, `models/` (per-model JSON, logs, comparison CSV), `figures/`, `tuning/` (Optuna trials), `hmm/`, `maps/` (Folium HTML + PNG) |
| `docs/` | method notes: preprocessing/features, model selection, HPO, HMM, AI-use disclosure |
| `tests/` | API contract tests |
| `report/` | technical report PDF |

## Installation
```
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```
Data (not in the repo): put the course dataset under `data_raw/` as
`data_raw/train/<trip>/<trip>.track|.nodes|.arcs|.route`, `data_raw/student_test/<trip>/...` and the 10 reserved trips in
`data_raw/evaluator_test/<trip>/...`.

## Reproduce everything (in order)
| Step | Command | Output |
|---|---|---|
| 1 Data audit + cleaning | `python src/preprocess.py` | `results/preprocessing/` |
| 2 EDA | `python src/eda.py` | `results/eda/` |
| 3 Candidates + features | `python src/build_cache.py` | `cache/` (~1 min) |
| 4 8-model comparison | `python src/train_all.py` | `results/models/`, `results/figures/` (~6 min) |
| 5 Bayesian optimisation | `python src/tune_bayes.py 40` | `results/tuning/`, `artifacts/lgbm_bayes.joblib` |
| 6 Final model | `python src/train_final.py` | `artifacts/`, `results/models/final_model_eval.json` |
| 7 HMM tuning + comparison | `python src/hmm_experiments.py` | `artifacts/decode_config.json`, `results/hmm/` |
| 8 Folium maps | `python src/folium_maps.py` then `python src/screenshot_maps.py` | `results/maps/` (interactive HTML is generated locally and not committed; PNG screenshots are committed) |
| 9 Report | `python src/make_report.py` | `report/PathSnap_Technical_Report.pdf` |

Seed 42 everywhere. Selection uses 5-fold GroupKFold by trip on the 80 training trips; student-test is used for
development comparison; the evaluator trips are used only by the API at evaluation time.

## Deployment (two laptops)
**Laptop 1 – API** (needs the repo, `artifacts/` and `data_raw/evaluator_test/`)
```
python api/main.py
```
It listens on `0.0.0.0:8000` and logs every request in this terminal (also in `api/logs/requests.log`).
Find the laptop's IPv4 address with `ipconfig`. Both laptops must be on the same network (a phone hotspot works).
Windows may ask whether to allow Python through the firewall on first start: allow it on private networks.
Quick check from laptop 2: open `http://<laptop1-ip>:8000/health` in a browser.

**Laptop 2 – UI** (needs only the `ui/` folder and Python; the map tiles need internet)
```
python ui/serve_ui.py --api http://<laptop1-ip>:8000
```
The browser opens the evaluator page. Pick any of the 10 trips. The UI sends the trip id to `POST /predict` and draws the GPS
fixes, the nearby road network, the predicted route (and optionally the classifier-only route) and shows the metrics.

### API
| Method | Path | Body / result |
|---|---|---|
| GET | `/health` | status, model file, HMM parameters |
| GET | `/trips` | `[{"trip_id", "n_fixes"}]` for the reserved trips |
| POST | `/predict` | `{"trip_id": "00000015"}` → GPS fixes, predicted arc per fix, predicted route geometry, candidate road geometry, metrics, timings |

Interactive docs: `http://<laptop1-ip>:8000/docs`. Tests: `python tests/test_api.py`.

## Integrity rules followed
* No trip appears in two partitions; the evaluator trips share no trip id or file content with train/student-test.
* Models, features, hyperparameters and HMM parameters were chosen using the 80 training trips (cross-validation) and
  the 10 student-test trips only. The evaluator trips are never used for training, tuning or selection.
* The prediction pipeline loads trips **without** the `.route` file (`pipeline.predict_trip`). The reference route is read
  only afterwards in `pipeline.score_trip` on the server, and only numeric metrics are returned. The UI never receives
  the reference route (checked in `tests/test_api.py`).
* No hard-coded predictions: every request runs the full pipeline (features → classifier → HMM).
* Reference labels per fix are *derived* by aligning fixes to the `.route` arcs (monotone DP); they are not
  independently verified ground truth (see `docs/PREPROCESSING_FEATURES.md`).

## AI assistance
See `docs/AI_USE.md`.
