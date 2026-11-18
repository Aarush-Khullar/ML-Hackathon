"""Train & compare 8 models: 5-fold group CV on the 80 training trips + evaluation on the 10 student-test trips.
Usage: python src/run_experiments.py [model_name ...]    -> results/models/*  (terminal output also saved to results/models/training_log.txt)"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from models_zoo import zoo
from data_utils import load_trip, list_trips
from decode import decode_trip, load_costs
from report import model_block, summary_table, Tee

STRIDE = 3  # train on every 3rd fix (1 Hz fixes are near-duplicates); evaluate on ALL fixes
OUT = os.path.join(ROOT, "results", "models"); os.makedirs(OUT, exist_ok=True)
sys.stdout = Tee(os.path.join(OUT, "training_log.txt"))
print(f"PathSnap model comparison  ({time.strftime('%Y-%m-%d %H:%M')}, seed {SEED})", flush=True)

tr, tr_info, tr_nfix, tr_nvalid = load_partition("train", True)
te, te_info, te_nfix, _ = load_partition("student_test", False)
Xtr, ytr, gtr = feature_matrix(tr), tr.y.values, tr.trip.values
Xte, yte = feature_matrix(te), te.y.values
sub = (tr.i.values % STRIDE == 0)
print(f"train: {len(tr):,} (fix, candidate) rows from 80 trips, fitted on every {STRIDE}rd fix = {sub.sum():,} rows; {tr_nvalid:,}/{tr_nfix:,} fixes with reliable label")
print(f"test : {len(te):,} rows, {te_nfix:,} fixes from 10 student-test trips; {Xtr.shape[1]} features", flush=True)

student_dir = os.path.join(ROOT, "data_raw", "student_test")
arcs_by = {t: load_trip(student_dir, t, False)["arcs"] for t in list_trips(student_dir)}
time_by = {t: load_trip(student_dir, t, False)["track"].t.values for t in list_trips(student_dir)}

gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(Xtr, ytr, gtr))
names = sys.argv[1:] or list(zoo().keys()); all_res = []
for name in names:
    fam, cfg, mk = zoo()[name]
    res = dict(model=name, family=fam, config=cfg, stride=STRIDE, seed=SEED)
    cv_f1, cv_top1, cv_ll = [], [], []; t0 = time.time()
    print(f"\n>>> training {name} ({fam}) ...", flush=True)
    for k, (a, b) in enumerate(folds):
        a = a[sub[a]]; m = mk(); m.fit(Xtr[a], ytr[a]); p = m.predict_proba(Xtr[b])[:, 1]
        met = candidate_metrics(ytr[b], p); dfb = tr.iloc[b]
        nfix_b = len(set(zip(dfb.trip.values, dfb.i.values)))
        cv_f1.append(met["macro_f1"]); cv_ll.append(met["log_loss"]); cv_top1.append(top1(dfb, p, nfix_b))
        print(f"    fold {k + 1}/5 done: macro-F1 {cv_f1[-1]:.4f}  Top-1 {cv_top1[-1]:.4f}", flush=True)
    res.update(cv_macro_f1=float(np.mean(cv_f1)), cv_macro_f1_std=float(np.std(cv_f1)), cv_top1=float(np.mean(cv_top1)),
               cv_top1_std=float(np.std(cv_top1)), cv_logloss=float(np.mean(cv_ll)), cv_time_s=time.time() - t0,
               cv_fold_macro_f1=[float(x) for x in cv_f1], cv_fold_top1=[float(x) for x in cv_top1], cv_fold_logloss=[float(x) for x in cv_ll])
    # final fit on all training trips -> student test
    t0 = time.time(); m = mk(); m.fit(Xtr[sub], ytr[sub]); res["train_time_s"] = time.time() - t0
    ptr = m.predict_proba(Xtr[sub][:300000])[:, 1]; res["train_macro_f1"] = candidate_metrics(ytr[sub][:300000], ptr)["macro_f1"]
    t0 = time.time(); p = m.predict_proba(Xte)[:, 1]; inf_t = time.time() - t0
    res["inference_s_per_1000_fixes"] = 1000 * inf_t / te_nfix
    met = candidate_metrics(yte, p); res.update({f"test_{k}": v for k, v in met.items()})
    res["test_top1"] = top1(te, p, te_nfix)
    pred = top1_arcs(te, p); res["test_continuity"] = continuity(pred, arcs_by, time_by)
    res["test_accuracy"] = float(((p >= 0.5) == yte).mean())
    ok = 0; costs = load_costs(ROOT)                       # Viterbi decoding of this model's probabilities
    for t in arcs_by:
        msk = te.trip.values == t; dec = decode_trip(te[msk], p[msk], arcs_by[t], costs)
        ok += int((dec.arc.values == te_info[t]["label_arc"][dec.i.values]).sum())
    res["test_top1_viterbi"] = ok / te_nfix
    from sklearn.metrics import confusion_matrix
    res["test_confusion"] = confusion_matrix(yte, (p >= 0.5).astype(int)).tolist()
    np.save(os.path.join(OUT, f"{name}_student_proba.npy"), p.astype(np.float32))
    if name == "M8_MLP":   # epoch history on 8 held-out TRAIN trips (never student/evaluator data)
        trips = sorted(set(gtr)); val_t = set(trips[::10][:8]); vm = np.isin(gtr, list(val_t))
        mm = mk(); mm.fit(Xtr[(~vm) & sub], ytr[(~vm) & sub], Xtr[vm][::3], ytr[vm][::3])
        pd.DataFrame(mm.history_).to_csv(os.path.join(OUT, "M8_MLP_history.csv"), index=False)
    json.dump(res, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    print(model_block(res), flush=True); all_res.append(res)
print(summary_table(all_res), flush=True)
