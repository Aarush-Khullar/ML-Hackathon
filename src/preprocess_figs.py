"""Figures + example tables for the preprocessing / feature-engineering evaluation."""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import clean, kinematic_features, ROOT, MAX_SPEED
from data_utils import load_trip, list_trips
from features import build_trip, _headings, FEATURES
OUT = os.path.join(ROOT, "results", "preprocessing")
d = load_trip(os.path.join(ROOT, "data_raw", "student_test"), "00000007", False)
k = kinematic_features(clean(d["track"]))
hx, hy, spd, valid = _headings(k.x.values, k.y.values, k.t.values)
smooth = (np.degrees(np.arctan2(hx, hy)) + 360) % 360
sp_all, turn_all, dt_all = [], [], []
for part in ["train", "student_test"]:
    for t in list_trips(os.path.join(ROOT, "data_raw", part)):
        f = kinematic_features(clean(load_trip(os.path.join(ROOT, "data_raw", part), t, False)["track"]))
        sp_all.append(f.speed.values[1:]); turn_all.append(f.turn_angle.values[1:]); dt_all.append(f.dt.values[1:])
sp_all, turn_all, dt_all = map(np.concatenate, (sp_all, turn_all, dt_all))
fig, ax = plt.subplots(2, 3, figsize=(15, 7))
s = slice(0, 300)
ax[0, 0].plot(k.t[s], k.speed[s], lw=.8, label="step speed (1-s differences)"); ax[0, 0].plot(k.t[s], spd[s], lw=1.5, label="smoothed (±3 fixes)"); ax[0, 0].set_title("Speed, trip 00000007 (student_test)", fontsize=10); ax[0, 0].set_xlabel("t (s)"); ax[0, 0].set_ylabel("m/s"); ax[0, 0].legend(fontsize=7)
ax[0, 1].plot(k.t[s], k.bearing[s], lw=.8, label="raw per-step bearing"); ax[0, 1].plot(k.t[s], smooth[s], lw=1.5, label="smoothed heading (±3 fixes)"); ax[0, 1].set_title("Bearing: noisy per-step vs smoothed", fontsize=10); ax[0, 1].set_ylabel("deg from north"); ax[0, 1].legend(fontsize=7)
ax[0, 2].hist(np.clip(dt_all, 0, 8), bins=np.arange(0, 8.5, .5), color="#4C72B0"); ax[0, 2].set_yscale("log"); ax[0, 2].set_title("Time gap between fixes (s), all 90 trips, clipped at 8", fontsize=10)
ax[1, 0].hist(np.clip(sp_all, 0, 70), bins=70, color="#55A868"); ax[1, 0].axvline(MAX_SPEED, c="r", ls="--"); ax[1, 0].set_yscale("log"); ax[1, 0].set_title(f"Step speed (m/s), red = {MAX_SPEED:.0f} m/s outlier flag", fontsize=10)
ax[1, 1].hist(turn_all, bins=72, color="#C44E52"); ax[1, 1].set_yscale("log"); ax[1, 1].set_title("Per-step turning angle (deg): GPS noise makes raw heading erratic", fontsize=9)
df, _ = build_trip(d, labels=False); g = df.groupby("i").size()
ax[1, 2].hist(df.dist, bins=60, color="#8172B2"); ax[1, 2].set_title(f"Candidate-arc distance to fix (m); {g.mean():.1f} candidates/fix", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_preprocessing.png"), dpi=130)
fix = int(df.i.iloc[len(df) // 2]); ex = df[df.i == fix][["i", "arc"] + FEATURES].round(3); ex.to_csv(os.path.join(OUT, "example_candidate_features_one_fix.csv"), index=False)
print(ex.head(5).T.to_string())
print("speed>50:", (sp_all > 50).sum(), "of", len(sp_all), "| step speed p50/p99", np.percentile(sp_all, [50, 99]).round(1))
