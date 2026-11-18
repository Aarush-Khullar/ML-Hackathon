"""FINAL RESULTS: train all 8 models on the 80 training trips, then test every model on BOTH held-out sets
(student_test and evaluator_test), plus the saved final model artifacts/lgbm_final.joblib.
Nothing is trained or tuned on student/evaluator data -- they are only predicted.
Usage: python src/final_results.py        (~2 min; needs cache/ from build_cache.py for all 3 partitions)
Writes: results/models/final_results_all_models.csv  and  results/models/final_results_log.txt"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, load_partition, feature_matrix, top1, candidate_metrics, final_model_path
from models_zoo import zoo
from data_utils import load_trip
from decode import decode_trip, load_costs
from report import Tee

OUT = os.path.join(ROOT, "results", "models"); STRIDE = 3
open(os.path.join(OUT, "final_results_log.txt"), "w").close(); sys.stdout = Tee(os.path.join(OUT, "final_results_log.txt"))
costs = load_costs(ROOT)
tr, _, _, _ = load_partition("train", True)
sub = tr.i.values % STRIDE == 0
Xtr, ytr = feature_matrix(tr)[sub], tr.y.values[sub]
TESTS = {}
for part in ("student_test", "evaluator_test"):
    df, info, nfix, _ = load_partition(part, False)
    arcs = {t: load_trip(os.path.join(ROOT, "data_raw", part), t, False)["arcs"] for t in info}
    TESTS[part] = (df, info, nfix, feature_matrix(df), arcs)
print("=" * 100)
print(f"PathSnap FINAL RESULTS  ({time.strftime('%Y-%m-%d %H:%M')})")
print(f"Training : 80 train trips, {len(Xtr):,} (fix, road) rows, {Xtr.shape[1]} features")
for part, (df, info, nfix, X, _) in TESTS.items():
    print(f"Test     : {part:15s} {len(info)} trips, {nfix:,} GPS fixes, {len(df):,} candidate rows  (never used for training/tuning)")
print("TRAINING uses ONLY data_raw/train. student_test and evaluator_test are used ONLY for prediction (testing).")
print(f"Saved final model and Viterbi costs are loaded as-is (final model created {time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(final_model_path())))}, never refitted here).")
print("=" * 100, flush=True)


def evaluate(model_name, p_fn):
    out = []
    for part, (df, info, nfix, X, arcs) in TESTS.items():
        t0 = time.time(); p = p_fn(X); inf = time.time() - t0; y = df.y.values; m = candidate_metrics(y, p)
        ok = 0
        for t in info:
            msk = df.trip.values == t; dec = decode_trip(df[msk], p[msk], arcs[t], costs)
            ok += int((dec.arc.values == info[t]["label_arc"][dec.i.values]).sum())
        r = dict(model=model_name, test_set=part, accuracy=float(((p >= .5) == y).mean()), macro_f1=m["macro_f1"],
                 precision=m["precision"], recall=m["recall"], log_loss=m["log_loss"],
                 top1=top1(df, p, nfix), top1_viterbi=ok / nfix, predict_s=inf)
        out.append(r)
        print(f"   {part:15s} accuracy {100 * r['accuracy']:6.2f} %  macro-F1 {100 * r['macro_f1']:6.2f} %  precision {100 * r['precision']:6.2f} %"
              f"  recall {100 * r['recall']:6.2f} %  Top-1 {100 * r['top1']:6.2f} %  Top-1+Viterbi {100 * r['top1_viterbi']:6.2f} %", flush=True)
    return out


rows = []
models = list(zoo().items())
for k, (name, (fam, cfg, mk)) in enumerate(models, 1):
    print(f"\n[{k}/{len(models) + 1}] {name}  ({fam}; {cfg})", flush=True)
    t0 = time.time(); m = mk(); m.fit(Xtr, ytr); print(f"   trained on 80 train trips in {time.time() - t0:.1f} s", flush=True)
    rows += evaluate(name, lambda X: m.predict_proba(X)[:, 1])
print(f"\n[{len(models) + 1}/{len(models) + 1}] FINAL saved model artifacts/lgbm_final.joblib  (Bayesian-tuned LightGBM)", flush=True)
fm = joblib.load(final_model_path())
rows += evaluate("LightGBM_FINAL", lambda X: fm.predict_proba(X)[:, 1])

R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "final_results_all_models.csv"), index=False)
for part in TESTS:
    T = R[R.test_set == part].sort_values("top1_viterbi", ascending=False)
    print("\n" + "=" * 100 + f"\nFINAL TABLE - {part.upper()}  ({TESTS[part][2]:,} GPS fixes, {len(TESTS[part][1])} unseen trips)   sorted by Top-1 + Viterbi")
    print("-" * 100)
    print(f"{'Model':18s}{'Accuracy':>10s}{'Macro-F1':>10s}{'Precision':>11s}{'Recall':>9s}{'Log loss':>10s}{'Top-1':>9s}{'Top-1+Viterbi':>15s}")
    print("-" * 100)
    for _, r in T.iterrows():
        print(f"{r.model:18s}{100 * r.accuracy:9.2f}%{100 * r.macro_f1:9.2f}%{100 * r.precision:10.2f}%{100 * r.recall:8.2f}%"
              f"{r.log_loss:10.4f}{100 * r.top1:8.2f}%{100 * r.top1_viterbi:14.2f}%")
C = R.groupby("model", sort=False)[["accuracy", "macro_f1", "top1", "top1_viterbi"]].mean().sort_values("top1_viterbi", ascending=False)
print("\n" + "=" * 100 + "\nCOMBINED (mean of student_test and evaluator_test)\n" + "-" * 100)
print(f"{'Model':18s}{'Accuracy':>10s}{'Macro-F1':>10s}{'Top-1':>9s}{'Top-1+Viterbi':>15s}")
for name, r in C.iterrows():
    print(f"{name:18s}{100 * r.accuracy:9.2f}%{100 * r.macro_f1:9.2f}%{100 * r.top1:8.2f}%{100 * r.top1_viterbi:14.2f}%")
f = R[R.model == "LightGBM_FINAL"].set_index("test_set")
print("\n" + "=" * 100 + "\nSELECTED MODEL: Bayesian-tuned LightGBM (artifacts/lgbm_final.joblib)")
for part in TESTS:
    r = f.loc[part]
    print(f"  {part:15s}: accuracy {100 * r.accuracy:.2f} %, macro-F1 {100 * r.macro_f1:.2f} %, Top-1 {100 * r.top1:.2f} %, Top-1 + Viterbi {100 * r.top1_viterbi:.2f} %")
print("Accuracy = every (GPS fix, candidate road) pair classified correctly; inflated because ~88 % of candidates are easy negatives.")
print("Top-1 = share of GPS fixes whose chosen road is correct (main metric). Viterbi = road-graph decoding of consecutive fixes.")
print("Saved: results/models/final_results_all_models.csv, results/models/final_results_log.txt\n" + "=" * 100)
