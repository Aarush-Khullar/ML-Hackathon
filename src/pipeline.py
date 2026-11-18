"""End-to-end inference pipeline used by the API and the Folium maps.

predict_trip(...)  : raw .track/.nodes/.arcs  -> candidates + features -> classifier -> HMM/Viterbi -> one arc per fix.
                     The reference .route is NOT read here (labels=False), so ground truth can never reach the model.
score_trip(...)    : server-side evaluation. Reads the reference .route separately, derives per-fix reference arcs
                     and compares them with the prediction. Only the resulting numbers leave this function.
"""
import os, sys, json, time
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import load_trip
from features import build_trip, align_labels, FEATURES
from hmm import load_config, decode_trip, argmax_trip, continuity, route_breaks, sequence_metrics, successor_nodes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PathSnapModel:
    def __init__(self, root=ROOT):
        self.conf = json.load(open(os.path.join(root, "artifacts", "model_config.json")))
        self.model = joblib.load(os.path.join(root, self.conf["model"]))
        self.hmm_cfg = load_config(root)
        self.R, self.K = float(self.conf.get("R", 60.0)), int(self.conf.get("K", 12))

    def predict_trip(self, part_dir, trip_id):
        t0 = time.time()
        d = load_trip(part_dir, trip_id, with_route=False)            # ground truth deliberately not loaded
        df, info = build_trip(d, R=self.R, K=self.K, labels=False)
        t1 = time.time()
        p = self.model.predict_proba(df[FEATURES].to_numpy(dtype=np.float32))[:, 1]
        t2 = time.time()
        n = len(d["track"])
        cls = argmax_trip(df, p, n)
        dec = decode_trip(df, p, d["arcs"], self.hmm_cfg, t=d["track"].t.values, n_fix=n)
        t3 = time.time()
        return dict(trip=d, cand=df, p=p, pred_classifier=cls.arc.values, pred_hmm=dec.arc.values,
                    timing=dict(features_s=t1 - t0, classifier_s=t2 - t1, hmm_s=t3 - t2, total_s=t3 - t0))


def score_trip(part_dir, trip_id, pred, d=None, cand=None):
    """Server-side scoring against the reference route (never returned to clients)."""
    full = load_trip(part_dir, trip_id, with_route=True)
    if full["route"] is None:
        return None
    lab, lab_dist = align_labels(full)
    reliable = lab_dist <= 60.0
    arcs, t = full["arcs"], full["track"].t.values; succ = successor_nodes(arcs)
    out = {}
    for name, a in (("classifier", pred["pred_classifier"]), ("hmm", pred["pred_hmm"])):
        out[f"top1_{name}"] = float((a == lab).mean())
        out[f"top1_{name}_reliable_fixes"] = float((a[reliable] == lab[reliable]).mean()) if reliable.any() else None
        out[f"continuity_{name}"] = continuity(a, arcs, t, succ=succ)
        out[f"breaks_{name}"] = route_breaks(a, arcs, succ)
        out.update({f"{k}_{name}": v for k, v in sequence_metrics(a, full["route"]).items()})
    if cand is not None:
        have = cand.groupby("i").arc.apply(set)
        out["candidate_miss_rate"] = float(np.mean([lab[i] not in have.get(i, set()) for i in range(len(lab))]))
    out["reliable_label_share"] = float(reliable.mean())
    return out


def route_geometry(d, arc_seq):
    """lat/lon polyline segments for a (compressed) sequence of arc ids."""
    n, a = d["nodes"], d["arcs"]; segs = []
    seq = [int(x) for x in arc_seq if x >= 0]; seq = [x for k, x in enumerate(seq) if k == 0 or x != seq[k - 1]]
    for x in seq:
        u, v = int(a.u.values[x]), int(a.v.values[x])
        segs.append(dict(arc=x, coords=[[float(n.lat.values[u]), float(n.lon.values[u])], [float(n.lat.values[v]), float(n.lon.values[v])]]))
    return segs
