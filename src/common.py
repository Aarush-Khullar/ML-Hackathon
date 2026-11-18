"""Shared loading + metric helpers used by experiments, tuning, HMM and API."""
import os, sys, pickle, time
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, confusion_matrix
sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42
R_VALID = 60.0   # a derived reference label further than this from its fix is treated as unreliable


def load_partition(part, drop_invalid_ref):
    df = pd.read_pickle(os.path.join(ROOT, "cache", f"{part}_cand.pkl"))
    info = pickle.load(open(os.path.join(ROOT, "cache", f"{part}_info.pkl"), "rb"))
    n_total = sum(v["n_fix"] for v in info.values())
    if drop_invalid_ref:
        bad = {t: set(np.where(v["label_dist"] > R_VALID)[0].tolist()) for t, v in info.items()}
        keep = np.fromiter((i not in bad[t] for t, i in zip(df.trip.values, df.i.values)), bool, len(df))
        df = df[keep].reset_index(drop=True)
        n_valid = n_total - sum(len(b) for b in bad.values())
    else:
        n_valid = n_total
    return df, info, n_total, n_valid


def top1(df, p, n_fix):
    """Observation-level Top-1 accuracy. Fixes whose reference arc is not among the candidates count as errors."""
    d = pd.DataFrame({"trip": df.trip.values, "i": df.i.values, "p": p, "y": df.y.values})
    idx = d.groupby(["trip", "i"], sort=False).p.idxmax()
    return float(d.loc[idx, "y"].sum() / n_fix)


def top1_arcs(df, p):
    """Return DataFrame(trip,i,arc) of the arg-max candidate per fix."""
    d = pd.DataFrame({"trip": df.trip.values, "i": df.i.values, "arc": df.arc.values, "p": p})
    idx = d.groupby(["trip", "i"], sort=False).p.idxmax()
    return d.loc[idx, ["trip", "i", "arc"]].reset_index(drop=True)


def candidate_metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    return dict(macro_f1=f1_score(y, yh, average="macro"), f1_pos=f1_score(y, yh, pos_label=1),
                precision=precision_score(y, yh, zero_division=0), recall=recall_score(y, yh, zero_division=0),
                log_loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)))


def continuity(pred, arcs_by_trip, time_by_trip, max_hops=2):
    """Share of consecutive 1-Hz-ish fix pairs whose predicted arcs are identical or reachable within <=max_hops arcs."""
    ok = tot = 0
    for trip, g in pred.groupby("trip", sort=False):
        a = arcs_by_trip[trip]; u, v = a.u.values, a.v.values
        out = {}
        for k, (x, y_) in enumerate(zip(u, v)): out.setdefault(x, []).append(k)
        t = time_by_trip[trip]
        g = g.sort_values("i"); I, A = g.i.values, g.arc.values
        for j in range(len(I) - 1):
            if I[j + 1] != I[j] + 1 or t[I[j + 1]] - t[I[j]] > 3: continue
            a0, a1 = A[j], A[j + 1]; tot += 1
            if a0 == a1 or u[a1] == v[a0]: ok += 1; continue
            if max_hops >= 2:
                if any(v[c] == u[a1] for c in out.get(v[a0], [])): ok += 1
    return ok / max(tot, 1)


def feature_matrix(df):
    return df[FEATURES].to_numpy(dtype=np.float32)


def final_model_path():
    """Path of the selected final model, as recorded by train_final.py in artifacts/model_config.json."""
    import json
    return os.path.join(ROOT, json.load(open(os.path.join(ROOT, "artifacts", "model_config.json")))["model"])
