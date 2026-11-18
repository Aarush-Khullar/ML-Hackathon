"""HMM / Viterbi experiments (reproducible):
 1. out-of-fold (5-fold GroupKFold) probabilities of the FINAL classifier on the 80 training trips
 2. decoder-parameter tuning by coordinate search on those TRAINING out-of-fold probabilities
    (objective: pointwise Top-1 after decoding) -> artifacts/decode_config.json
 3. student-test comparison: classifier-only (argmax) vs HMM without gap handling vs full HMM
    pointwise Top-1, route continuity, route breaks, route precision/recall/F1 and LCS vs reference route
The evaluator-test partition is not used.
Usage : python src/hmm_experiments.py [--skip-tuning]
Writes: results/hmm/{decoder_tuning.csv, hmm_comparison_per_trip.csv, hmm_summary.csv, hmm_log.txt, fig_hmm_*.png}
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib, lightgbm as lgb
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import GroupKFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, SEED, load_partition, feature_matrix
from data_utils import load_trip, list_trips
from hmm import DEFAULTS, load_config, decode_trip, argmax_trip, continuity, route_breaks, sequence_metrics, successor_nodes
from report import Tee

OUT = os.path.join(ROOT, "results", "hmm")


def _decode_one(args):
    d, t, arcs, cfg, lab = args
    dec = decode_trip(d, d.pp.values, arcs, cfg, t=t, n_fix=len(t))
    keep = lab >= 0
    return int((dec.arc.values[keep] == lab[keep]).sum()), int(keep.sum())


def score_train(groups, cfg, ex):
    r = list(ex.map(_decode_one, [(d, t, a, cfg, l) for d, t, a, l in groups], chunksize=2))
    return sum(x for x, _ in r) / sum(y for _, y in r)


def main():
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "hmm_log.txt"), "w").close(); sys.stdout = Tee(os.path.join(OUT, "hmm_log.txt"))
    conf = json.load(open(os.path.join(ROOT, "artifacts", "model_config.json")))
    params = {k: v for k, v in conf["params"].items() if k not in ("random_state", "n_jobs", "verbose")}
    print(f"HMM experiments with final classifier {conf['model']} params {params}", flush=True)
    cfg = load_config(ROOT)

    if "--skip-tuning" not in sys.argv:
        # ---- 1. training out-of-fold probabilities
        tr, info, _, _ = load_partition("train", True)
        X, y, g = feature_matrix(tr), tr.y.values, tr.trip.values; sub = tr.i.values % 3 == 0
        oof = np.zeros(len(tr)); t0 = time.time()
        for a, b in GroupKFold(5).split(X, y, g):
            a = a[sub[a]]
            m = lgb.LGBMClassifier(**params, random_state=SEED, n_jobs=16, verbose=-1).fit(X[a], y[a])
            oof[b] = m.predict_proba(X[b])[:, 1]
        print(f"training OOF probabilities in {time.time() - t0:.0f} s", flush=True)
        tr = tr.assign(pp=oof)[["trip", "i", "arc", "u", "v", "pp"]]
        tdir = os.path.join(ROOT, "data_raw", "train"); groups = []
        for t, d in tr.groupby("trip", sort=False):
            raw = load_trip(tdir, t, False); lab = info[t]["label_arc"].copy()
            lab[info[t]["label_dist"] > 60.0] = -1                      # unreliable labels are not scored
            groups.append((d, raw["track"].t.values, raw["arcs"], lab))
        # ---- 2. coordinate search on decoder parameters (training data only)
        grid = dict(c_twin=[5, 10, 20], c_jump=[30, 60, 100], c_2hop=[2, 3, 5], c_next=[0.5, 1, 2], alpha=[0.5, 0.7, 1.0],
                    c_hop=[0.5, 1, 2], max_hops=[4, 8, 12])
        rows = []
        with ProcessPoolExecutor(16) as ex:
            best_cfg = dict(cfg); best = score_train(groups, best_cfg, ex)
            rows.append(dict(step="start", **best_cfg, train_top1=best)); print(f"start config train Top-1 {best:.4f}", flush=True)
            for k, vals in grid.items():
                for val in vals:
                    c = dict(best_cfg); c[k] = val
                    if c == best_cfg: continue
                    s = score_train(groups, c, ex); rows.append(dict(step=f"{k}={val}", **c, train_top1=s))
                    print(f"  {k:9s}= {val:<6} train Top-1 {s:.4f}{'  <- better' if s > best + 1e-5 else ''}", flush=True)
                    if s > best + 1e-5: best, best_cfg = s, c
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "decoder_tuning.csv"), index=False)
        best_cfg = {k: (float(v) if k != "max_hops" else int(v)) for k, v in best_cfg.items() if k in DEFAULTS}
        json.dump(dict(best_cfg, tuned_on="out-of-fold probabilities of the final classifier on the 80 TRAINING trips "
                                        "(5-fold GroupKFold); coordinate search; objective pointwise Top-1 after decoding",
                       train_oof_top1_decoded=best), open(os.path.join(ROOT, "artifacts", "decode_config.json"), "w"), indent=1)
        print(f"best decoder config {best_cfg}  train OOF Top-1 {best:.4f}", flush=True)
        cfg = best_cfg

    # ---- 3. student-test comparison with the saved final classifier
    model = joblib.load(os.path.join(ROOT, conf["model"]))
    te, info, _, _ = load_partition("student_test", False)
    p_all = model.predict_proba(feature_matrix(te))[:, 1]; te = te.assign(pp=p_all)
    sdir = os.path.join(ROOT, "data_raw", "student_test"); rows = []
    for t in list_trips(sdir):
        raw = load_trip(sdir, t, True); arcs, tt, n = raw["arcs"], raw["track"].t.values, len(raw["track"])
        d = te[te.trip == t]; lab = info[t]["label_arc"]; succ = successor_nodes(arcs)
        variants = {"classifier_only": argmax_trip(d, d.pp.values, n),
                    "hmm_no_gap": decode_trip(d, d.pp.values, arcs, cfg, t=None, n_fix=n),
                    "hmm_full": decode_trip(d, d.pp.values, arcs, cfg, t=tt, n_fix=n)}
        for name, dec in variants.items():
            a = dec.arc.values
            rows.append(dict(trip=t, method=name, fixes=n, top1=float((a == lab).mean()), continuity=continuity(a, arcs, tt),
                             breaks=route_breaks(a, arcs, succ), gaps=int((np.diff(tt) > cfg["gap_s"]).sum()),
                             **sequence_metrics(a, raw["route"])))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "hmm_comparison_per_trip.csv"), index=False)
    w = R.assign(ok=R.top1 * R.fixes)
    S = w.groupby("method", sort=False).agg(fixes=("fixes", "sum"), ok=("ok", "sum"), continuity=("continuity", "mean"),
                                           breaks=("breaks", "sum"), route_precision=("route_precision", "mean"),
                                           route_recall=("route_recall", "mean"), route_f1=("route_f1", "mean"),
                                           route_lcs_ratio=("route_lcs_ratio", "mean"))
    S.insert(1, "top1", S.ok / S.fixes); S = S.drop(columns="ok"); S.to_csv(os.path.join(OUT, "hmm_summary.csv"))
    print("\nSTUDENT-TEST: classifier-only vs HMM (pointwise + sequence level)")
    print(S.round(4).to_string())
    print("\nper trip Top-1:"); print(R.pivot(index="trip", columns="method", values="top1").round(4).to_string())

    piv = R.pivot(index="trip", columns="method", values="top1")[["classifier_only", "hmm_no_gap", "hmm_full"]]
    ax = piv.plot.bar(figsize=(10, 4.5), width=.8); ax.set_ylim(.75, 1); ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Student-test: classifier-only vs HMM/Viterbi per trip"); plt.xticks(rotation=30); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_hmm_per_trip.png"), dpi=130); plt.close()
    m = S[["top1", "continuity", "route_precision", "route_recall", "route_f1"]]
    ax = m.T.plot.bar(figsize=(9, 4.5), width=.8); ax.set_ylim(.5, 1.0); ax.set_title("Student-test: pointwise and sequence-level metrics")
    plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_hmm_summary.png"), dpi=130); plt.close()
    print("saved -> results/hmm/")


if __name__ == "__main__":
    main()
