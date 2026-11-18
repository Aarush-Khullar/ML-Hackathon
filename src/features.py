"""Candidate generation, reference-label alignment and feature engineering for PathSnap.

Formulation: every GPS fix is a "query"; every directed road arc within R metres is a "candidate".
The classifier predicts P(candidate is the arc the vehicle is on | features); Top-1 per fix = argmax.
"""
import os, sys
import numpy as np, pandas as pd
import shapely
from shapely.strtree import STRtree
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_trip

FEATURES = ["dist", "frac", "proj_out", "arc_len", "cos_hd", "abs_ang", "speed", "hd_valid",
            "dist_rank", "dist_gap", "dist_ratio", "n_cand", "ctx_dist_mean", "ctx_dist_max",
            "ctx_prog", "out_deg_v", "in_deg_u", "has_twin",
            "same_prev", "same_next", "conn_prev", "conn_next", "dmin_end",
            # long-range context (added for v2): wide-window heading + distance/progress over +-5/+-10/+-20 fixes
            "cos_hd_w10", "hd_valid_w10", "cos_hd_w20", "ctx_dist_mean_w10", "ctx_dist_max_w10",
            "ctx_prog_w10", "ctx_prog_w20", "dist_z", "cos_hd_rank"]


def arc_geometry(d):
    n, a = d["nodes"], d["arcs"]
    ax_, ay_ = n.x.values[a.u.values], n.y.values[a.u.values]
    bx_, by_ = n.x.values[a.v.values], n.y.values[a.v.values]
    return ax_, ay_, bx_, by_


def point_seg(px, py, ax_, ay_, bx_, by_):
    """Distance, fraction (unclipped), of points to segments (broadcast-friendly)."""
    dx, dy = bx_ - ax_, by_ - ay_
    L2 = np.maximum(dx * dx + dy * dy, 1e-9)
    t = ((px - ax_) * dx + (py - ay_) * dy) / L2
    tc = np.clip(t, 0, 1)
    cx, cy = ax_ + tc * dx, ay_ + tc * dy
    return np.hypot(px - cx, py - cy), t


def align_labels(d, lam=0.3, hd_pen=0.0):
    """Monotone DP alignment of each GPS fix to an arc of the reference route.
    The .route file only lists arcs (no fix->arc map), so a per-fix reference label is *derived*:
    minimise sum of (point-to-arc distance + heading-disagreement penalty) subject to a non-decreasing
    route position. label_dist returned is the pure geometric distance of the fix to its aligned arc."""
    tr, r = d["track"], d["route"]
    ax_, ay_, bx_, by_ = arc_geometry(d)
    px, py = tr.x.values[:, None], tr.y.values[:, None]
    C, _ = point_seg(px, py, ax_[r][None], ay_[r][None], bx_[r][None], by_[r][None])  # N x K
    # Optional heading-aware cost (experiment, OFF by default: hd_pen=20 lowered CV Top-1 0.906 -> 0.904 and
    # student Top-1 0.922 -> 0.914 because smoothed heading lags at turns). Twin arcs of a two-way road have identical geometry, so distance alone
    # cannot tell which direction the route is using at that moment. Add a penalty (up to HD_PEN m)
    # when the smoothed GPS heading points against the arc direction (only when the vehicle is moving).
    hx, hy, _, mv = _headings(tr.x.values, tr.y.values, tr.t.values, w=5, min_disp=5.0)
    L = np.maximum(np.hypot(bx_[r] - ax_[r], by_[r] - ay_[r]), 1e-9)
    cos = (hx[:, None] * ((bx_[r] - ax_[r]) / L)[None] + hy[:, None] * ((by_[r] - ay_[r]) / L)[None])
    C = C + hd_pen * 0.5 * (1.0 - cos) * mv[:, None]
    C = np.minimum(C, 200.0)
    N, K = C.shape
    ks = np.arange(K)
    f = C[0].copy()
    back = np.zeros((N, K), dtype=np.int32)
    for i in range(1, N):
        g = f - lam * ks
        pm = np.minimum.accumulate(g)
        # argmin of prefix
        isnew = g <= pm
        idx = np.where(isnew, ks, 0); idx = np.maximum.accumulate(idx)
        back[i] = idx
        f = C[i] + pm + lam * ks
    k = int(np.argmin(f)); path = np.zeros(N, dtype=np.int32)
    for i in range(N - 1, -1, -1):
        path[i] = k; k = back[i, k]
    D, _ = point_seg(px, py, ax_[r][None], ay_[r][None], bx_[r][None], by_[r][None])
    return r[path], D[np.arange(N), path]   # label_dist stays a pure geometric distance


def _headings(x, y, t, w=3, min_disp=4.0):
    n = len(x); i = np.arange(n)
    lo, hi = np.clip(i - w, 0, n - 1), np.clip(i + w, 0, n - 1)
    dx, dy, dt = x[hi] - x[lo], y[hi] - y[lo], np.maximum(t[hi] - t[lo], 1e-3)
    dist = np.hypot(dx, dy); speed = dist / dt
    valid = dist > min_disp
    hx, hy = np.where(valid, dx, 0.0), np.where(valid, dy, 0.0)
    # forward fill invalid headings with last valid one
    last = -1; hx2, hy2 = hx.copy(), hy.copy()
    for j in range(n):
        if valid[j]: last = j
        elif last >= 0: hx2[j], hy2[j] = hx[last], hy[last]
    nxt = -1                       # backward fill: stationary fixes at the very start take the first valid heading
    for j in range(n - 1, -1, -1):
        if valid[j]: nxt = j
        elif nxt >= 0 and np.hypot(hx2[j], hy2[j]) == 0: hx2[j], hy2[j] = hx[nxt], hy[nxt]
    nrm = np.maximum(np.hypot(hx2, hy2), 1e-9)
    return hx2 / nrm, hy2 / nrm, speed, valid


def build_trip(d, R=60.0, K=12, labels=True):
    """Return candidate DataFrame (one row per (fix, arc)) and per-fix info."""
    tr, a, nodes = d["track"], d["arcs"], d["nodes"]
    ax_, ay_, bx_, by_ = arc_geometry(d)
    A = len(a); N = len(tr); Nn = len(nodes)
    px, py, pt = tr.x.values, tr.y.values, tr.t.values
    lines = shapely.linestrings(np.stack([np.stack([ax_, ay_], 1), np.stack([bx_, by_], 1)], 1))
    tree = STRtree(lines)
    pts = shapely.points(px, py)
    qi, ai = tree.query(pts, predicate="dwithin", distance=R)
    df = pd.DataFrame({"i": qi.astype(np.int32), "arc": ai.astype(np.int32)})
    # fixes with no candidate: widen to nearest arc within 300 m
    have = np.zeros(N, bool); have[df.i.values] = True
    miss = np.where(~have)[0]
    if len(miss):
        q2, a2 = tree.query_nearest(pts[miss], max_distance=300.0, all_matches=False)
        df = pd.concat([df, pd.DataFrame({"i": miss[q2].astype(np.int32), "arc": a2.astype(np.int32)})], ignore_index=True)
    ii, aa = df.i.values, df.arc.values
    dist, t = point_seg(px[ii], py[ii], ax_[aa], ay_[aa], bx_[aa], by_[aa])
    df["dist"] = dist
    df = df.sort_values(["i", "dist"], kind="stable")
    df["dist_rank"] = df.groupby("i").cumcount()
    df = df[df.dist_rank < K].reset_index(drop=True)
    ii, aa = df.i.values, df.arc.values
    dist, t = point_seg(px[ii], py[ii], ax_[aa], ay_[aa], bx_[aa], by_[aa])
    df["dist"] = dist.astype(np.float32)
    df["frac"] = np.clip(t, 0, 1).astype(np.float32)
    df["proj_out"] = np.maximum(np.maximum(-t, t - 1), 0).astype(np.float32)
    alen = np.hypot(bx_ - ax_, by_ - ay_)
    df["arc_len"] = alen[aa].astype(np.float32)
    df["dmin_end"] = np.minimum(np.hypot(px[ii] - ax_[aa], py[ii] - ay_[aa]), np.hypot(px[ii] - bx_[aa], py[ii] - by_[aa])).astype(np.float32)
    # heading
    hx, hy, speed, valid = _headings(px, py, pt)
    adx, ady = (bx_ - ax_) / np.maximum(alen, 1e-9), (by_ - ay_) / np.maximum(alen, 1e-9)
    cosd = np.clip(hx[ii] * adx[aa] + hy[ii] * ady[aa], -1, 1)
    df["cos_hd"] = (cosd * valid[ii]).astype(np.float32)
    df["abs_ang"] = (np.arccos(cosd) * valid[ii]).astype(np.float32)
    df["speed"] = speed[ii].astype(np.float32); df["hd_valid"] = valid[ii].astype(np.float32)
    # ranking features
    g = df.groupby("i").dist
    dmin = g.transform("min").values
    df["dist_gap"] = (df.dist.values - dmin).astype(np.float32)
    df["dist_ratio"] = (df.dist.values / (dmin + 1.0)).astype(np.float32)
    df["n_cand"] = g.transform("size").values.astype(np.float32)
    df["dist_z"] = ((df.dist.values - g.transform("mean").values) / (g.transform("std").fillna(0).values + 1.0)).astype(np.float32)
    # temporal context: distance of this arc to fixes i-3..i+3 and along-arc progress
    cd = []
    for off in (-3, -2, -1, 1, 2, 3):
        j = np.clip(ii + off, 0, N - 1)
        dj, _ = point_seg(px[j], py[j], ax_[aa], ay_[aa], bx_[aa], by_[aa]); cd.append(np.minimum(dj, 100.0))
    cd = np.stack(cd, 1)
    df["ctx_dist_mean"] = cd.mean(1).astype(np.float32); df["ctx_dist_max"] = cd.max(1).astype(np.float32)
    jl, jh = np.clip(ii - 3, 0, N - 1), np.clip(ii + 3, 0, N - 1)
    prog = ((px[jh] - px[jl]) * adx[aa] + (py[jh] - py[jl]) * ady[aa])
    df["ctx_prog"] = np.clip(prog, -60, 60).astype(np.float32)
    # long-range context: heading over +-10 / +-20 fixes, distance to fixes +-5..+-10, progress over +-10 / +-20
    for w, md in ((10, 8.0), (20, 15.0)):
        hxw, hyw, _, vw = _headings(px, py, pt, w=w, min_disp=md)
        cw = np.clip(hxw[ii] * adx[aa] + hyw[ii] * ady[aa], -1, 1)
        df[f"cos_hd_w{w}"] = (cw * (vw[ii] | (np.hypot(hxw, hyw)[ii] > 0))).astype(np.float32)
        if w == 10: df["hd_valid_w10"] = vw[ii].astype(np.float32)
    cd = []
    for off in (-10, -7, -5, 5, 7, 10):
        j = np.clip(ii + off, 0, N - 1)
        dj, _ = point_seg(px[j], py[j], ax_[aa], ay_[aa], bx_[aa], by_[aa]); cd.append(np.minimum(dj, 150.0))
    cd = np.stack(cd, 1)
    df["ctx_dist_mean_w10"] = cd.mean(1).astype(np.float32); df["ctx_dist_max_w10"] = cd.max(1).astype(np.float32)
    for w, c in ((10, 200), (20, 400)):
        jl, jh = np.clip(ii - w, 0, N - 1), np.clip(ii + w, 0, N - 1)
        df[f"ctx_prog_w{w}"] = np.clip((px[jh] - px[jl]) * adx[aa] + (py[jh] - py[jl]) * ady[aa], -c, c).astype(np.float32)
    # rank of heading agreement among the candidates of the same fix (0 = best aligned)
    df["cos_hd_rank"] = df.groupby("i").cos_hd_w10.rank(ascending=False, method="min").sub(1).astype(np.float32)
    # topology
    u, v = d["arcs"].u.values, d["arcs"].v.values
    df["u"], df["v"] = u[aa].astype(np.int32), v[aa].astype(np.int32)   # arc end nodes (used by the stage-2 context model)
    out_deg = np.bincount(u, minlength=Nn); in_deg = np.bincount(v, minlength=Nn)
    df["out_deg_v"] = out_deg[v[aa]].astype(np.float32); df["in_deg_u"] = in_deg[u[aa]].astype(np.float32)
    twin = set(zip(v.tolist(), u.tolist()))
    df["has_twin"] = np.array([(u[x], v[x]) in twin for x in range(A)], dtype=np.float32)[aa] if A < 400000 else 0.0
    # temporal connectivity with neighbouring candidate sets
    keyA = ii.astype(np.int64) * A + aa
    prev_same = np.isin((ii - 1).astype(np.int64) * A + aa, keyA)
    next_same = np.isin((ii + 1).astype(np.int64) * A + aa, keyA)
    end_nodes = np.unique(ii.astype(np.int64) * Nn + v[aa])      # (fix, end node of candidate)
    start_nodes = np.unique(ii.astype(np.int64) * Nn + u[aa])    # (fix, start node of candidate)
    conn_prev = np.isin((ii - 1).astype(np.int64) * Nn + u[aa], end_nodes)
    conn_next = np.isin((ii + 1).astype(np.int64) * Nn + v[aa], start_nodes)
    df["same_prev"], df["same_next"] = prev_same.astype(np.float32), next_same.astype(np.float32)
    df["conn_prev"], df["conn_next"] = conn_prev.astype(np.float32), conn_next.astype(np.float32)
    info = dict(n_fix=N)
    if labels and d["route"] is not None:
        lab_arc, lab_dist = align_labels(d)
        df["y"] = (aa == lab_arc[ii]).astype(np.int8)
        info["label_arc"] = lab_arc; info["label_dist"] = lab_dist
        info["miss_rate"] = float(1 - df.groupby("i").y.max().reindex(range(N), fill_value=0).mean())
    df.insert(0, "trip", d["trip_id"])
    return df, info
