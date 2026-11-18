"""Aggregate results/models/*.json -> model_comparison_full.csv and draw the comparison figures.
Usage: python src/make_figures.py      (after run_experiments.py and train_final.py)"""
import os, sys, json, glob
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, final_model_path
from features import FEATURES
M = os.path.join(ROOT, "results", "models"); F = os.path.join(ROOT, "results", "figures"); os.makedirs(F, exist_ok=True)
rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(M, "M*.json")))]
df = pd.DataFrame(rows); conf = df.pop("test_confusion") if "test_confusion" in df else None
fin = json.load(open(os.path.join(M, "final_model_eval.json")))
df.to_csv(os.path.join(M, "model_comparison_full.csv"), index=False)
cols = ["model", "cv_macro_f1", "cv_macro_f1_std", "cv_top1", "test_accuracy", "test_macro_f1", "test_top1", "test_top1_viterbi", "test_log_loss", "train_time_s"]
print(df[cols].round(4).to_string(index=False))
print(f"\nFINAL saved model (artifacts/lgbm_final.joblib): candidate acc {fin['test_candidate_accuracy']:.4f}, macro-F1 {fin['test_macro_f1']:.4f}, "
      f"Top-1 {fin['test_top1_argmax']:.4f}, Top-1+Viterbi {fin['test_top1_decoded']:.4f}")
names = [m.split("_", 1)[1] for m in df.model]; x = np.arange(len(df))

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x, df.cv_macro_f1, yerr=df.cv_macro_f1_std, capsize=3, color="#4C72B0")
ax.set_xticks(x, names, rotation=20); ax.set_ylim(0.75, 1); ax.set_ylabel("macro-F1"); ax.set_title("5-fold trip-grouped CV macro-F1 (80 training trips)")
for k, v in enumerate(df.cv_macro_f1): ax.text(k, v + 0.03, f"{v:.3f}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(F, "fig_cv_f1.png"), dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(10, 4.5)); w = 0.2
for j, (c, lbl) in enumerate([("test_accuracy", "candidate accuracy"), ("test_macro_f1", "macro-F1"),
                              ("test_top1", "Top-1 (argmax)"), ("test_top1_viterbi", "Top-1 (+Viterbi)")]):
    ax.bar(x + (j - 1.5) * w, df[c], w, label=lbl)
ax.axhline(0.96, c="k", ls="--", lw=.8); ax.set_xticks(x, names, rotation=20); ax.set_ylim(0.75, 1); ax.legend(fontsize=8, ncol=4, loc="lower center")
ax.set_title("Student-test (10 unseen trips, 19,160 fixes)"); plt.tight_layout(); plt.savefig(os.path.join(F, "fig_test_top1_f1.png"), dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(9, 4)); ax.bar(x, df.test_log_loss, color="#C44E52"); ax.set_xticks(x, names, rotation=20); ax.set_yscale("log")
ax.set_title("Student-test log loss (lower = better calibrated probabilities -> better HMM emissions)"); plt.tight_layout(); plt.savefig(os.path.join(F, "fig_logloss.png"), dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(9, 4)); ax.bar(x - .2, df.train_macro_f1, .4, label="train"); ax.bar(x + .2, df.cv_macro_f1, .4, label="CV")
ax.set_xticks(x, names, rotation=20); ax.set_ylim(0.75, 1); ax.legend(); ax.set_title("Over/under-fitting: train vs CV macro-F1"); plt.tight_layout(); plt.savefig(os.path.join(F, "fig_overfit_gap.png"), dpi=130); plt.close()

if conf is not None:
    fig, axs = plt.subplots(2, 4, figsize=(14, 6.5))
    for a_, n_, c_ in zip(axs.ravel(), names, conf):
        c_ = np.array(c_); a_.imshow(c_, cmap="Blues"); a_.set_title(n_, fontsize=9); a_.set_xticks([0, 1], ["pred 0", "pred 1"]); a_.set_yticks([0, 1], ["true 0", "true 1"])
        for (r, k), v in np.ndenumerate(c_): a_.text(k, r, f"{v:,}", ha="center", va="center", fontsize=8, color="w" if v > c_.max() / 2 else "k")
    plt.tight_layout(); plt.savefig(os.path.join(F, "fig_confusion_matrices.png"), dpi=130); plt.close()

h = os.path.join(M, "M8_MLP_history.csv")
if os.path.exists(h):
    hh = pd.read_csv(h); fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(hh.epoch, hh.train_loss, label="train"); ax.plot(hh.epoch, hh.val_loss, label="validation (held-out train trips)")
    ax.set_xlabel("epoch"); ax.set_ylabel("log loss"); ax.legend(); ax.set_title("MLP learning curve"); plt.tight_layout(); plt.savefig(os.path.join(F, "fig_mlp_loss.png"), dpi=130); plt.close()

m = joblib.load(final_model_path())
g = m.booster_.feature_importance("gain"); g = g / g.sum(); o = np.argsort(g)
fig, ax = plt.subplots(figsize=(7, 8)); ax.barh(np.array(FEATURES)[o], g[o], color="#55A868"); ax.set_xlabel("gain share")
ax.set_title("Final LightGBM feature importance (gain)"); plt.tight_layout(); plt.savefig(os.path.join(F, "fig_feature_importance.png"), dpi=130); plt.close()
pd.DataFrame({"feature": np.array(FEATURES)[o[::-1]], "gain_share": g[o[::-1]]}).to_csv(os.path.join(M, "final_feature_importance.csv"), index=False)
print("figures ->", F)
