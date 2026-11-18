"""Live demo for evaluation 1: load the saved model and predict road candidates for one trip.
Usage: python src/demo_model.py [trip]      (student_test trip id, default 00000076)"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_trip
from features import build_trip, FEATURES
from decode import decode_trip, load_costs
from common import ROOT, final_model_path
trip = sys.argv[1] if len(sys.argv) > 1 else "00000076"
model = joblib.load(final_model_path())
d = load_trip(os.path.join(ROOT, "data_raw", "student_test"), trip, True)
df, info = build_trip(d)                       # candidates + 32 features (+ derived labels, used for scoring only)
n = info["n_fix"]; lab = info["label_arc"]
t0 = time.time(); df["p"] = model.predict_proba(df[FEATURES].to_numpy(dtype=np.float32))[:, 1]; dt = time.time() - t0
top = df.loc[df.groupby("i").p.idxmax()]
t1 = time.time(); dec = decode_trip(df, df.p.values, d["arcs"], load_costs(ROOT)); dt2 = time.time() - t1
print(f"trip {trip}: {n} fixes, {len(df)} candidate rows ({len(df) / n:.1f}/fix), model inference {dt:.3f}s, Viterbi {dt2:.2f}s")
print(f"candidate-level accuracy            {100 * ((df.p >= .5) == df.y).mean():.2f}%")
print(f"Top-1 road accuracy (per-fix argmax) {100 * (top.arc.values == lab[top.i.values]).sum() / n:.2f}%")
print(f"Top-1 road accuracy (+Viterbi)       {100 * (dec.arc.values == lab[dec.i.values]).sum() / n:.2f}%")
print(f"candidate miss rate                  {100 * info['miss_rate']:.2f}%\n")
print(top[["i", "arc", "dist", "cos_hd", "ctx_prog", "p"]].head(8).round(3).to_string(index=False))
