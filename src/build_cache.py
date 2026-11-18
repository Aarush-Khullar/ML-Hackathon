"""Build candidate/feature tables for train and student_test partitions -> cache/*.pkl"""
import os, sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_trip, list_trips
from features import build_trip
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(ROOT, "cache"), exist_ok=True)
R, K = 60.0, 12
# usage: python src/build_cache.py [partition ...]   (default: every partition present in data_raw/)
PARTS = sys.argv[1:] or [x for x in ["train", "student_test", "evaluator_test"] if os.path.isdir(os.path.join(ROOT, "data_raw", x))]
for part in PARTS:
    pdir = os.path.join(ROOT, "data_raw", part); dfs, infos = [], {}
    for tid in list_trips(pdir):
        d = load_trip(pdir, tid); df, info = build_trip(d, R=R, K=K); dfs.append(df); infos[tid] = info
    big = pd.concat(dfs, ignore_index=True); big.to_pickle(os.path.join(ROOT, "cache", f"{part}_cand.pkl"))
    pickle.dump(infos, open(os.path.join(ROOT, "cache", f"{part}_info.pkl"), "wb"))
    nf = sum(i["n_fix"] for i in infos.values()); miss = sum(i["miss_rate"] * i["n_fix"] for i in infos.values()) / nf
    print(part, "rows", len(big), "fixes", nf, "pos_rate", round(big.y.mean(), 4), "candidate_miss_rate", round(miss, 4))
