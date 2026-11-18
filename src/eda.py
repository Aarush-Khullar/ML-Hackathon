"""Exploratory data analysis. Usage: python src/eda.py  (writes results/eda/*)"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_trip, list_trips

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "eda"); os.makedirs(OUT, exist_ok=True)
PARTS = {"train": os.path.join(ROOT, "data_raw", "train"),
         "student_test": os.path.join(ROOT, "data_raw", "student_test")}

rows = []
for part, pdir in PARTS.items():
    for tid in list_trips(pdir):
        d = load_trip(pdir, tid)
        tr, n, a, r = d["track"], d["nodes"], d["arcs"], d["route"]
        dt = np.diff(tr.t.values)
        step = np.hypot(np.diff(tr.x.values), np.diff(tr.y.values))
        N = len(n)
        g = coo_matrix((np.ones(len(a)), (a.u.values, a.v.values)), shape=(N, N))
        ncomp, lab = connected_components(g, directed=False)
        deg = np.bincount(np.r_[a.u.values, a.v.values], minlength=N)
        route_ok = int(sum(a.v.values[r[i]] == a.u.values[r[i + 1]] for i in range(len(r) - 1))) if r is not None else -1
        rows.append(dict(
            partition=part, trip=tid, n_gps=len(tr), n_nodes=N, n_arcs=len(a), route_len=len(r),
            nan_gps=int(tr[["lon", "lat", "t"]].isna().sum().sum()),
            files=len(os.listdir(os.path.join(pdir, tid))),
            kb_total=round(sum(os.path.getsize(os.path.join(pdir, tid, f)) for f in os.listdir(os.path.join(pdir, tid))) / 1024),
            dt_exact1_pct=round(100 * np.mean(dt == 1), 2), dt_max=float(dt.max()), n_dt_gt5=int((dt > 5).sum()),
            t_nonmonotonic=int((dt <= 0).sum()),
            step_median_m=round(float(np.median(step)), 2), step_max_m=round(float(step.max()), 1),
            n_step_gt100m=int((step > 100).sum()),
            extent_km=round(float(np.hypot(np.ptp(tr.x), np.ptp(tr.y)) / 1000), 2),
            net_components=int(ncomp), largest_comp_pct=round(100 * np.bincount(lab).max() / N, 2),
            isolated_nodes=int((deg == 0).sum()), self_loops=int((a.u.values == a.v.values).sum()),
            route_connected_pairs=route_ok, route_pairs=len(r) - 1,
            route_arcs_in_range=bool((r >= 0).all() and (r < len(a)).all()),
            lon0=round(d["origin"][0], 3), lat0=round(d["origin"][1], 3)))
inv = pd.DataFrame(rows)
inv.to_csv(os.path.join(OUT, "dataset_inventory.csv"), index=False)
summ = inv.groupby("partition").agg(trips=("trip", "count"), gps_total=("n_gps", "sum"), gps_med=("n_gps", "median"),
        arcs_med=("n_arcs", "median"), nodes_med=("n_nodes", "median"), route_med=("route_len", "median"),
        nan=("nan_gps", "sum"), dt_gt5=("n_dt_gt5", "sum"), nonmono=("t_nonmonotonic", "sum"),
        step_gt100=("n_step_gt100m", "sum"), multi_comp_trips=("net_components", lambda s: int((s > 1).sum())),
        isolated=("isolated_nodes", "sum"), disc_routes=("route_connected_pairs", lambda s: 0)).reset_index()
disc = inv.assign(d=inv.route_pairs - inv.route_connected_pairs).groupby("partition").d.sum()
summ["route_disconnected_pairs"] = summ.partition.map(disc)
summ.drop(columns="disc_routes").to_csv(os.path.join(OUT, "partition_summary.csv"), index=False)
print(summ.drop(columns="disc_routes").T.to_string())

# ---- plots
tr_inv = inv[inv.partition == "train"]
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].hist(inv.n_gps, bins=30, color="#4C72B0"); ax[0].set_title("GPS observations per trip (train+student)"); ax[0].set_xlabel("# points")
ax[1].hist(inv.extent_km, bins=30, color="#55A868"); ax[1].set_title("Trajectory extent per trip"); ax[1].set_xlabel("km")
ax[2].scatter(inv.n_arcs, inv.n_gps, c=(inv.partition == "train").map({True: "#4C72B0", False: "#C44E52"}), s=14)
ax[2].set_xscale("log"); ax[2].set_title("Road-network size vs GPS points"); ax[2].set_xlabel("# arcs"); ax[2].set_ylabel("# GPS")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_trip_sizes.png"), dpi=130); plt.close()

alldt, allstep = [], []
for part, pdir in PARTS.items():
    for tid in list_trips(pdir):
        t = load_trip(pdir, tid, with_route=False)["track"]
        alldt.append(np.diff(t.t.values)); allstep.append(np.hypot(np.diff(t.x.values), np.diff(t.y.values)))
alldt, allstep = np.concatenate(alldt), np.concatenate(allstep)
vc = pd.Series(alldt).round(0).value_counts().sort_index()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].bar(vc.index[:12].astype(str), vc.values[:12], color="#4C72B0"); ax[0].set_yscale("log"); ax[0].set_title("Time gap between consecutive fixes (s)")
ax[1].hist(np.clip(allstep, 0, 60), bins=60, color="#55A868"); ax[1].set_yscale("log"); ax[1].set_title("Step length between fixes (m, clipped 60)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_dt_step.png"), dpi=130); plt.close()
stats = dict(pct_dt_eq_1=float(100 * np.mean(alldt == 1)), pct_dt_le_2=float(100 * np.mean(alldt <= 2)), max_dt=float(alldt.max()),
             step_p50=float(np.median(allstep)), step_p99=float(np.percentile(allstep, 99)), step_max=float(allstep.max()),
             n_fixes=int(len(alldt) + len(inv)))
json.dump(stats, open(os.path.join(OUT, "global_stats.json"), "w"), indent=1); print(stats)

# trajectory plot with network + route for 3 trips
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for ax, tid in zip(axs, ["00000003", "00000005", "00000008"]):
    d = load_trip(PARTS["train"], tid); n, a, r, tr = d["nodes"], d["arcs"], d["route"], d["track"]
    m = (n.x.values[a.u] > tr.x.min() - 300) & (n.x.values[a.u] < tr.x.max() + 300) & (n.y.values[a.u] > tr.y.min() - 300) & (n.y.values[a.u] < tr.y.max() + 300)
    for u, v in zip(a.u.values[m], a.v.values[m]):
        ax.plot([n.x[u], n.x[v]], [n.y[u], n.y[v]], c="#BBBBBB", lw=0.4)
    for k in r: ax.plot([n.x[a.u[k]], n.x[a.v[k]]], [n.y[a.u[k]], n.y[a.v[k]]], c="#2ca02c", lw=1.6)
    ax.scatter(tr.x, tr.y, s=2, c="#1f77b4"); ax.set_aspect("equal"); ax.set_title(f"train trip {tid}: GPS (blue), reference route (green), network (grey)", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_trajectories.png"), dpi=130); plt.close()
print("done")

# ---- missing-value / data-quality chart (train + student_test)
q = inv.groupby("partition")[["nan_gps", "t_nonmonotonic", "n_dt_gt5", "n_step_gt100m"]].sum()
q.columns = ["missing values", "non-monotonic t", "gaps > 5 s", "jumps > 100 m"]
ax = q.T.plot.bar(figsize=(8, 4), color=["#C44E52", "#4C72B0"]); ax.set_ylabel("count (log)"); ax.set_yscale("symlog")
ax.set_title("Data-quality checks per partition (0 missing values, 0 unordered timestamps)"); plt.xticks(rotation=0)
for c in ax.containers: ax.bar_label(c, fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_missing_quality.png"), dpi=130); plt.close()

# ---- road-network connectivity (train + student_test)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
ax[0].hist(inv.net_components, bins=30, color="#8172B2"); ax[0].set_title("Weakly connected components per trip network"); ax[0].set_xlabel("# components")
ax[1].hist(inv.largest_comp_pct, bins=30, color="#55A868"); ax[1].set_title("Share of nodes in the largest component"); ax[1].set_xlabel("%")
d = load_trip(PARTS["train"], "00000001", False); a = d["arcs"]
deg = np.bincount(np.r_[a.u.values, a.v.values], minlength=len(d["nodes"]))
ax[2].bar(*np.unique(np.clip(deg, 0, 10), return_counts=True), color="#4C72B0"); ax[2].set_yscale("log")
ax[2].set_title("Node degree distribution (train trip 00000001)"); ax[2].set_xlabel("in+out degree (clipped 10)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_road_network.png"), dpi=130); plt.close()
print("extra EDA figures done")
