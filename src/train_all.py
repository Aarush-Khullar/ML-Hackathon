"""Train all 8 models ONE BY ONE (each in its own process), start to finish, then the final saved model.
Every model's per-fold CV scores and student-test metrics are printed to the terminal as they finish,
and everything is also written to results/models/training_log.txt.
Usage: python src/train_all.py            (~10-12 min; needs cache/ from build_cache.py)"""
import os, sys, time, subprocess
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT
from models_zoo import zoo
PY, SRC = sys.executable, os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(ROOT, "results", "models", "training_log.txt")
os.makedirs(os.path.dirname(LOG), exist_ok=True)
open(LOG, "w").close()                                   # fresh log for this run
env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
names = list(zoo().keys()); T0 = time.time()
for k, name in enumerate(names, 1):
    print(f"\n######## [{k}/{len(names)}] {name} ########", flush=True)
    t0 = time.time()
    r = subprocess.run([PY, os.path.join(SRC, "run_experiments.py"), name], env=env)
    if r.returncode: sys.exit(f"{name} failed (exit {r.returncode})")
    print(f"######## {name} finished in {time.time() - t0:.0f} s ########", flush=True)
print("\n######## FINAL MODEL: training artifacts/lgbm_final.joblib ########", flush=True)
subprocess.run([PY, os.path.join(SRC, "train_final.py")], env=env, check=True)
subprocess.run([PY, os.path.join(SRC, "make_figures.py")], env=env, check=True, stdout=subprocess.DEVNULL)
print("\n######## ALL 8 MODELS ########", flush=True)
subprocess.run([PY, os.path.join(SRC, "show_results.py")], env=env, check=True)
print(f"\nTotal time {(time.time() - T0) / 60:.1f} min. Log: results/models/training_log.txt, figures: results/figures/", flush=True)
