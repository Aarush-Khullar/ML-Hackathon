"""HMM + Viterbi route reconstruction for PathSnap.

Hidden states   : for GPS fix i, the candidate road arcs of that fix (the same <=12 arcs within 60 m that the
                  classifier scored). A state keeps its arc_id, so the candidate -> road-segment link is preserved.
Emission score  : alpha * log P(arc | fix) from the classifier (LightGBM probability, floored at 1e-4).
Transition cost : (negative log-transition) between the arc at fix i-1 and the arc at fix i
                    same arc ................................ 0
                    successor (starts where previous ends) .. c_next
                    two-hop (one arc in between) ............ c_2hop
                    twin arc (same road, other direction) ... c_twin   (U-turn)
                    no connection ........................... c_jump   ("reset": lets the route break)
GPS gaps        : if the time step dt > gap_s, the vehicle may have driven several arcs while unobserved, so the
                  successor test is replaced by a hop-limited graph search (BFS up to max_hops arcs); cost =
                  c_next + c_hop * (hops - 1). Unreachable pairs still get c_jump.
Missing candidates: a fix with no candidate arc (nothing within 300 m) is not a state column; it is returned
                  with arc = -1 ("unmatched") and the chain continues from the previous fix.
Disconnected    : c_jump is finite, so when consecutive fixes only have unconnected candidates the decoder
                  pays a large cost and starts a new route piece; these "breaks" are counted and reported.
Decoding        : standard Viterbi recurrence in log space
                    V_i(s) = em_i(s) + max_r [ V_{i-1}(r) - cost(r -> s) ],   back-pointer = argmax_r
                  then back-tracking from argmax_s V_N(s).
Decoder parameters are tuned on training-trip out-of-fold predictions only (artifacts/decode_config.json).
"""
import json, os
from collections import deque
import numpy as np
import pandas as pd

DEFAULTS = dict(c_next=1.0, c_2hop=3.0, c_twin=10.0, c_jump=60.0, alpha=0.7, gap_s=5.0, c_hop=1.0, max_hops=8)


def load_config(root):
    p = os.path.join(root, "artifacts", "decode_config.json")
    cfg = dict(DEFAULTS)
    if os.path.exists(p):
        cfg.update({k: v for k, v in json.load(open(p)).items() if k in DEFAULTS})
    return cfg


def successor_nodes(arcs):
    out = {}
    for x, y in zip(arcs.u.values, arcs.v.values):
        out.setdefault(int(x), set()).add(int(y))
    return out


def _hops(succ, src, targets, max_hops):
    """BFS over directed road graph: minimum number of arcs from node src to each node in targets (<= max_hops)."""
    targets = set(targets); found = {}; seen = {src: 0}; q = deque([src])
    if src in targets: found[src] = 0
    while q and len(found) < len(targets):
        n = q.popleft(); d = seen[n]
        if d >= max_hops: continue
        for m in succ.get(n, ()):
            if m not in seen:
                seen[m] = d + 1; q.append(m)
                if m in targets: found[m] = d + 1
    return found


def viterbi(i, arc, u, v, p, t, succ, cfg):
    """Flat arrays (one row per fix-candidate pair) + per-fix timestamps t (indexed by fix id).
    Returns fix ids (sorted), chosen arc per fix, and per-step transition type."""
    o = np.lexsort((-p, i)); i, arc, u, v, p = i[o], arc[o], u[o], v[o], p[o]
    cuts = np.flatnonzero(np.diff(i)) + 1; starts, ends = np.r_[0, cuts], np.r_[cuts, len(i)]
    em = cfg["alpha"] * np.log(np.clip(p, 1e-4, 1.0))
    back, score, prev, prev_fix = [], None, None, None
    for s, e in zip(starts, ends):
        fx = int(i[s]); a, uu, vv = arc[s:e], u[s:e], v[s:e]
        if prev is None:
            score = em[s:e].copy(); back.append(None)
        else:
            pa, pu, pv = prev
            dt = float(t[fx] - t[prev_fix]) if t is not None else 1.0
            T = np.full((len(pa), len(a)), -cfg["c_jump"])
            if dt > cfg["gap_s"]:                                    # GPS gap: multi-hop reachability
                for r, x in enumerate(pv):
                    h = _hops(succ, int(x), set(uu.tolist()), int(cfg["max_hops"]))
                    for c, y in enumerate(uu):
                        if int(y) in h: T[r, c] = -(cfg["c_next"] + cfg["c_hop"] * h[int(y)])
            else:
                T[np.array([[y in succ.get(int(x), ()) for y in uu] for x in pv])] = -cfg["c_2hop"]
                T[pv[:, None] == uu[None]] = -cfg["c_next"]
            T[(pu[:, None] == vv[None]) & (pv[:, None] == uu[None])] = -cfg["c_twin"]
            T[pa[:, None] == a[None]] = 0.0
            M = score[:, None] + T
            b = M.argmax(0); score = M[b, np.arange(len(a))] + em[s:e]; back.append(b)
        prev, prev_fix = (a, uu, vv), fx
    k = int(score.argmax()); res = np.empty(len(starts), np.int64)
    for j in range(len(starts) - 1, -1, -1):
        res[j] = arc[starts[j] + k]
        if back[j] is not None: k = back[j][k]
    return i[starts], res


def decode_trip(df, p, arcs, cfg, t=None, n_fix=None):
    """df: candidate table of one trip (i, arc, u, v). Returns DataFrame(i, arc) with one row per fix 0..n_fix-1;
    fixes without candidates get arc = -1."""
    fx, res = viterbi(df.i.values, df.arc.values, df.u.values, df.v.values, np.asarray(p, float),
                      None if t is None else np.asarray(t, float), successor_nodes(arcs), cfg)
    n = n_fix if n_fix is not None else int(fx.max()) + 1
    out = np.full(n, -1, np.int64); out[fx] = res
    return pd.DataFrame({"i": np.arange(n), "arc": out})


def argmax_trip(df, p, n_fix):
    """Classifier-only prediction: highest-probability candidate per fix (arc = -1 if none)."""
    d = pd.DataFrame({"i": df.i.values, "arc": df.arc.values, "p": p})
    top = d.loc[d.groupby("i").p.idxmax()]
    out = np.full(n_fix, -1, np.int64); out[top.i.values] = top.arc.values
    return pd.DataFrame({"i": np.arange(n_fix), "arc": out})


# ----------------------------------------------------------------------------- evaluation helpers
def continuity(pred_arc, arcs, t, max_dt=3.0, succ=None):
    """Share of consecutive fix pairs (dt <= max_dt) whose predicted arcs are equal or reachable within <= 2 arcs
    (same definition as the 8-model comparison in common.continuity)."""
    u, v = arcs.u.values, arcs.v.values; ok = tot = 0
    succ = succ if succ is not None else successor_nodes(arcs)
    for j in range(len(pred_arc) - 1):
        a0, a1 = pred_arc[j], pred_arc[j + 1]
        if a0 < 0 or a1 < 0 or t[j + 1] - t[j] > max_dt: continue
        tot += 1; ok += (a0 == a1) or (v[a0] == u[a1]) or (u[a1] in succ.get(int(v[a0]), ()))
    return ok / max(tot, 1)


def route_breaks(pred_arc, arcs, succ):
    """Number of places where consecutive distinct predicted arcs are not connected within 2 arcs."""
    u, v = arcs.u.values, arcs.v.values; seq = [a for a in pred_arc if a >= 0]
    seq = [a for k, a in enumerate(seq) if k == 0 or a != seq[k - 1]]; br = 0
    for a0, a1 in zip(seq[:-1], seq[1:]):
        if v[a0] == u[a1] or any(m == u[a1] for m in succ.get(int(v[a0]), ())): continue
        br += 1
    return br


def sequence_metrics(pred_arc, ref_route):
    """Route-level comparison of the predicted arc sequence with the reference route (set based + LCS)."""
    seq = [int(a) for a in pred_arc if a >= 0]; seq = [a for k, a in enumerate(seq) if k == 0 or a != seq[k - 1]]
    ref = [int(a) for a in ref_route]; P, R = set(seq), set(ref); inter = len(P & R)
    prec, rec = inter / max(len(P), 1), inter / max(len(R), 1)
    # longest common subsequence of the two arc sequences (order-aware agreement)
    n, m = len(seq), len(ref); prevrow = [0] * (m + 1)
    for x in seq:
        cur = [0] * (m + 1)
        for j, yv in enumerate(ref, 1):
            cur[j] = prevrow[j - 1] + 1 if x == yv else max(prevrow[j], cur[j - 1])
        prevrow = cur
    return dict(route_precision=prec, route_recall=rec, route_f1=2 * prec * rec / max(prec + rec, 1e-9),
                route_lcs_ratio=prevrow[m] / max(m, 1), pred_route_len=len(seq), ref_route_len=len(ref))
