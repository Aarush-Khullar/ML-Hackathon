"""Train the FINAL candidate-road model (LightGBM, 32 features) on all 80 training trips, save it, and
evaluate it on the 10 student-test trips -- both per-fix arg-max and with temporal (Viterbi) decoding.

Usage:  python src/train_final.py          (needs cache/ from build_cache.py; ~1 min)
Writes: artifacts/lgbm_final.joblib, artifacts/lgbm_final.txt (text dump, not committed), artifacts/model_config.json,
        results/models/final_model_eval.json, results/models/final_per_trip.csv
"""
import os, sys, json, time
import numpy as np, pandas as pd, joblib, lightgbm as lgb
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, SEED, load_partition, feature_matrix, top1, candidate_metrics
from features import FEATURES
from data_utils import load_trip, list_trips
from decode import decode_trip, load_costs

STRIDE = 3
# Final configuration = best trial of the Bayesian optimisation (src/tune_bayes.py, selected by CV macro-F1 on the
# 80 training trips). Falls back to the untuned LightGBM default if the search has not been run.
BP = os.path.join(ROOT, "results", "tuning", "best_params.json")
best = json.load(open(BP))["best_params"] if os.path.exists(BP) else dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=20)
PARAMS = dict(best, subsample_freq=1, random_state=SEED, n_jobs=8, verbose=-1)
MODEL_FILE = "lgbm_final.joblib"

tr, _, _, _ = load_partition("train", True)
te, te_info, te_nfix, _ = load_partition("student_test", False)
sub = tr.i.values % STRIDE == 0
t0 = time.time()
model = lgb.LGBMClassifier(**PARAMS).fit(feature_matrix(tr)[sub], tr.y.values[sub])
train_s = time.time() - t0
os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
joblib.dump(model, os.path.join(ROOT, "artifacts", MODEL_FILE))
model.booster_.save_model(os.path.join(ROOT, "artifacts", MODEL_FILE.replace(".joblib", ".txt")))
json.dump(dict(model=f"artifacts/{MODEL_FILE}", source="Bayesian-tuned LightGBM (results/tuning/best_params.json)", features=FEATURES, R=60.0, K=12, stride=STRIDE, params=PARAMS),
          open(os.path.join(ROOT, "artifacts", "model_config.json"), "w"), indent=1)

p = model.predict_proba(feature_matrix(te))[:, 1]
met = candidate_metrics(te.y.values, p)
costs = load_costs(ROOT)
sd = os.path.join(ROOT, "data_raw", "student_test"); rows = []; ok_arg = ok_dec = 0
te = te.assign(p=p)
for t in list_trips(sd):
    d = te[te.trip == t]; n = te_info[t]["n_fix"]; lab = te_info[t]["label_arc"]
    arg = d.loc[d.groupby("i").p.idxmax()]
    a_ok = int((arg.arc.values == lab[arg.i.values]).sum())
    dec = decode_trip(d, d.p.values, load_trip(sd, t, False)["arcs"], costs)
    d_ok = int((dec.arc.values == lab[dec.i.values]).sum())
    ok_arg += a_ok; ok_dec += d_ok
    rows.append(dict(trip=t, fixes=n, top1_argmax=a_ok / n, top1_decoded=d_ok / n, candidate_miss=te_info[t]["miss_rate"]))
res = dict(model="LightGBM final (Bayesian-tuned, 32 features)", params=PARAMS, train_time_s=train_s,
           test_candidate_accuracy=float(((p >= 0.5) == te.y.values).mean()),
           test_top1_argmax=ok_arg / te_nfix, test_top1_decoded=ok_dec / te_nfix,
           **{f"test_{k}": v for k, v in met.items()}, decode_costs=costs)
os.makedirs(os.path.join(ROOT, "results", "models"), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, "results", "models", "final_model_eval.json"), "w"), indent=1)
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "models", "final_per_trip.csv"), index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
print(f"\nFINAL MODEL on student-test ({te_nfix} fixes):")
print(f"  candidate-level accuracy        : {100*res['test_candidate_accuracy']:.2f} %")
print(f"  candidate-level macro-F1        : {100*met['macro_f1']:.2f} %")
print(f"  Top-1 road accuracy (per fix)   : {100*res['test_top1_argmax']:.2f} %")
print(f"  Top-1 road accuracy (+Viterbi)  : {100*res['test_top1_decoded']:.2f} %")
