"""Accuracy test of the saved model on a held-out partition (prints to terminal, saves results).
Usage: python src/eval_student.py [partition]     partition = student_test (default) or evaluator_test
Builds features from the raw files (no cache needed) -> LightGBM probabilities -> argmax and Viterbi decoding.
Nothing is fitted or tuned here: the saved model and Viterbi costs are used exactly as trained."""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_trip, list_trips
from features import build_trip, FEATURES
from decode import decode_trip, load_costs
from common import ROOT, candidate_metrics, final_model_path
model = joblib.load(final_model_path()); costs = load_costs(ROOT)
part = sys.argv[1] if len(sys.argv) > 1 else "student_test"
sd = os.path.join(ROOT, "data_raw", part); ys, ps, rows = [], [], []; tot = ok_a = ok_d = 0
print(f"{'trip':10s}{'fixes':>7s}{'Top-1 argmax':>14s}{'Top-1 +Viterbi':>16s}{'cand.miss':>11s}")
for t in list_trips(sd):
    d = load_trip(sd, t, True); df, info = build_trip(d); lab = info["label_arc"]; n = info["n_fix"]
    p = model.predict_proba(df[FEATURES].to_numpy(dtype=np.float32))[:, 1]
    top = df.assign(p=p).loc[lambda x: x.groupby("i").p.idxmax()]
    a = int((top.arc.values == lab[top.i.values]).sum())
    dec = decode_trip(df, p, d["arcs"], costs); b = int((dec.arc.values == lab[dec.i.values]).sum())
    tot += n; ok_a += a; ok_d += b; ys.append(df.y.values); ps.append(p)
    rows.append(dict(trip=t, fixes=n, top1_argmax=a / n, top1_viterbi=b / n, candidate_miss=info["miss_rate"]))
    print(f"{t:10s}{n:7d}{100 * a / n:12.2f}%{100 * b / n:14.2f}%{100 * info['miss_rate']:10.2f}%")
y, p = np.concatenate(ys), np.concatenate(ps); m = candidate_metrics(y, p)
print(f"\nOVERALL {part} ({tot} fixes, {len(rows)} trips)")
print(f"  candidate-level accuracy            : {100 * ((p >= .5) == y).mean():.2f} %")
print(f"  candidate-level macro-F1            : {100 * m['macro_f1']:.2f} %")
print(f"  precision / recall (on-road class)  : {100 * m['precision']:.2f} % / {100 * m['recall']:.2f} %")
print(f"  log loss                            : {m['log_loss']:.4f}")
print(f"  Top-1 road accuracy (argmax)        : {100 * ok_a / tot:.2f} %")
print(f"  Top-1 road accuracy (+Viterbi)      : {100 * ok_d / tot:.2f} %")
res = dict(partition=part, trips=len(rows), fixes=tot, candidate_accuracy=float(((p >= .5) == y).mean()),
           **{f"candidate_{k}": float(v) for k, v in m.items()}, top1_argmax=ok_a / tot, top1_viterbi=ok_d / tot)
out = os.path.join(ROOT, "results", "models"); os.makedirs(out, exist_ok=True)
json.dump(res, open(os.path.join(out, f"eval_{part}.json"), "w"), indent=1)
pd.DataFrame(rows).to_csv(os.path.join(out, f"eval_{part}_per_trip.csv"), index=False)
print(f"saved -> results/models/eval_{part}.json, eval_{part}_per_trip.csv")
