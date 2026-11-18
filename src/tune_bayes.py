"""Bayesian hyperparameter optimisation of the selected model (LightGBM) with Optuna (TPE sampler).

Objective : mean macro-F1 over 5-fold GroupKFold (grouped by trip) on the 80 TRAINING trips only.
            (same selection metric as the 8-model comparison; fit on every 3rd fix, scored on all fixes)
Pruning   : MedianPruner on the running fold mean, to stop clearly bad trials early.
After the search the best configuration is refitted on all 80 training trips and compared with the
UNTUNED LightGBM (M6 default config) on identical data: CV macro-F1, student-test Top-1, macro-F1,
log loss and inference time. The evaluator-test partition is never loaded here.

Usage : python src/tune_bayes.py [n_trials=40]
Writes: results/tuning/optuna_trials.csv, best_params.json, tuned_vs_untuned.csv, tuning_log.txt,
        fig_optuna_history.png, fig_optuna_importance.png;  artifacts/lgbm_bayes.joblib
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib, optuna, lightgbm as lgb
from sklearn.model_selection import GroupKFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, SEED, load_partition, feature_matrix, top1, candidate_metrics
from report import Tee

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
STRIDE = 3
OUT = os.path.join(ROOT, "results", "tuning"); os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "tuning_log.txt"), "w").close(); sys.stdout = Tee(os.path.join(OUT, "tuning_log.txt"))

tr, _, _, _ = load_partition("train", True)
te, _, te_nfix, _ = load_partition("student_test", False)
X, y, g = feature_matrix(tr), tr.y.values, tr.trip.values
sub = tr.i.values % STRIDE == 0
folds = [(a[sub[a]], b) for a, b in GroupKFold(n_splits=5).split(X, y, g)]
print(f"Bayesian optimisation of LightGBM: {N_TRIALS} trials, TPE sampler (seed {SEED}), 5-fold GroupKFold on 80 training trips", flush=True)

SPACE = {  # name: (low, high, log-scale?)  -- justified in docs/HPO.md
    "n_estimators": (150, 900, False), "learning_rate": (0.01, 0.2, True), "num_leaves": (15, 255, True),
    "min_child_samples": (10, 200, True), "subsample": (0.5, 1.0, False), "colsample_bytree": (0.4, 1.0, False),
    "reg_lambda": (1e-3, 10.0, True), "reg_alpha": (1e-3, 10.0, True)}


def params_from(trial):
    p = {}
    for k, (lo, hi, lg) in SPACE.items():
        p[k] = trial.suggest_int(k, lo, hi, log=lg) if isinstance(lo, int) else trial.suggest_float(k, lo, hi, log=lg)
    return p


def make(p):
    return lgb.LGBMClassifier(**p, subsample_freq=1, random_state=SEED, n_jobs=16, verbose=-1)


def objective(trial):
    p = params_from(trial); f1s = []
    for k, (a, b) in enumerate(folds):
        m = make(p).fit(X[a], y[a]); f1s.append(candidate_metrics(y[b], m.predict_proba(X[b])[:, 1])["macro_f1"])
        trial.report(float(np.mean(f1s)), k)
        if trial.should_prune(): raise optuna.TrialPruned()
    trial.set_user_attr("fold_f1", [float(v) for v in f1s])
    return float(np.mean(f1s))


def log_cb(study, t):
    v = f"{t.value:.4f}" if t.value is not None else "pruned"
    print(f"  trial {t.number:3d}  {t.state.name:8s} CV macro-F1 {v}   best so far {study.best_value:.4f} (trial {study.best_trial.number})", flush=True)


t0 = time.time()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1), study_name="lgbm_pathsnap")
study.enqueue_trial({"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20,
                     "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1e-3, "reg_alpha": 1e-3})  # trial 0 = untuned default
study.optimize(objective, n_trials=N_TRIALS, callbacks=[log_cb])
search_s = time.time() - t0
df = study.trials_dataframe(); df.to_csv(os.path.join(OUT, "optuna_trials.csv"), index=False)
best = study.best_params
print(f"\nsearch finished in {search_s / 60:.1f} min; {len(study.trials)} trials "
      f"({sum(t.state.name == 'PRUNED' for t in study.trials)} pruned). best CV macro-F1 {study.best_value:.4f}")
print("best params:", json.dumps(best, indent=1))


def full_eval(name, p, cv_f1):
    t0 = time.time(); m = make(p).fit(X[sub], y[sub]); fit_s = time.time() - t0
    t0 = time.time(); pt = m.predict_proba(feature_matrix(te))[:, 1]; inf = time.time() - t0
    met = candidate_metrics(te.y.values, pt)
    return m, dict(model=name, cv_macro_f1=cv_f1, student_top1=top1(te, pt, te_nfix), student_macro_f1=met["macro_f1"],
                   student_log_loss=met["log_loss"], student_accuracy=float(((pt >= .5) == te.y.values).mean()),
                   train_time_s=fit_s, inference_ms_per_1000_fixes=1000 * 1000 * inf / te_nfix)


untuned = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20}
cv_untuned = study.trials[0].value
_, r0 = full_eval("LightGBM untuned (M6 default)", dict(untuned, subsample=1.0, colsample_bytree=1.0, reg_lambda=0.0, reg_alpha=0.0), cv_untuned)
mt, r1 = full_eval("LightGBM Bayesian-tuned", best, study.best_value)
cmp_ = pd.DataFrame([r0, r1]); cmp_.to_csv(os.path.join(OUT, "tuned_vs_untuned.csv"), index=False)
print("\nTUNED vs UNTUNED (same 80 training trips, same 10 student-test trips)")
print(cmp_.round(4).to_string(index=False))
joblib.dump(mt, os.path.join(ROOT, "artifacts", "lgbm_bayes.joblib"))
json.dump(dict(best_params=best, best_cv_macro_f1=study.best_value, untuned_cv_macro_f1=cv_untuned, n_trials=len(study.trials),
               search_time_s=search_s, space={k: list(v) for k, v in SPACE.items()}, sampler="TPE", pruner="MedianPruner",
               objective="mean macro-F1, 5-fold GroupKFold on 80 training trips"),
          open(os.path.join(OUT, "best_params.json"), "w"), indent=1)

v = [t.value for t in study.trials if t.value is not None]; n = [t.number for t in study.trials if t.value is not None]
fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(n, v, "o", ms=4, label="completed trial"); ax.plot(n, np.maximum.accumulate(v), "r-", label="best so far")
ax.axhline(cv_untuned, c="k", ls="--", lw=.8, label="untuned (trial 0)"); ax.set_xlabel("trial"); ax.set_ylabel("CV macro-F1"); ax.legend()
ax.set_title("Optuna TPE optimisation history (LightGBM)"); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_optuna_history.png"), dpi=130); plt.close()
try:
    imp = optuna.importance.get_param_importances(study)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.barh(list(imp)[::-1], list(imp.values())[::-1], color="#4C72B0"); ax.set_xlabel("importance (fANOVA)")
    ax.set_title("Hyperparameter importance"); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_optuna_importance.png"), dpi=130); plt.close()
except Exception as e:
    print("importance plot skipped:", e)
print("saved -> results/tuning/, artifacts/lgbm_bayes.joblib")
