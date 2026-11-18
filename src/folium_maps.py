"""Interactive Folium maps generated from the actual pipeline.

  python src/folium_maps.py                 -> default set below
  python src/folium_maps.py student_test 00000076

Map types
  eda_<part>_<trip>.html     : GPS observations + road network (arcs within 150 m of the track) + reference route
  result_<part>_<trip>.html  : the same + classifier-only route + HMM/Viterbi predicted route (from pipeline.py)
Every map has distinguishable colours, a legend, layer control, start/end markers and is zoomed to the trip.
Outputs go to results/maps/. Maps are only made for train / student_test trips (development data).
"""
import os, sys
import numpy as np, folium
import shapely
from shapely.strtree import STRtree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import load_trip
from features import arc_geometry
from pipeline import PathSnapModel, score_trip, ROOT

OUT = os.path.join(ROOT, "results", "maps")
COL = dict(road="#98a2b3", gps="#1570ef", ref="#12b76a", cls="#f79009", hmm="#d92d20")


def _legend(items):
    rows = "".join(f'<div><span style="display:inline-block;width:22px;height:{h}px;background:{c};margin-right:6px;vertical-align:middle;'
                   f'border-radius:{r}"></span>{t}</div>' for t, c, h, r in items)
    return folium.Element(f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 12px;'
                          f'border:1px solid #ccc;border-radius:6px;font:13px sans-serif;line-height:20px">{rows}</div>')


def _arc_line(d, x):
    n, a = d["nodes"], d["arcs"]; u, v = a.u.values[x], a.v.values[x]
    return [(n.lat.values[u], n.lon.values[u]), (n.lat.values[v], n.lon.values[v])]


def _route_layer(d, seq, name, color, weight, dash=None, show=True):
    fg = folium.FeatureGroup(name=name, show=show)
    seq = [int(x) for x in seq if x >= 0]; seq = [x for k, x in enumerate(seq) if k == 0 or x != seq[k - 1]]
    for x in seq:
        folium.PolyLine(_arc_line(d, x), color=color, weight=weight, opacity=.9, dash_array=dash, tooltip=f"{name}: arc {x}").add_to(fg)
    return fg


def base_map(d, title):
    tr = d["track"]; m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])
    m.get_root().html.add_child(folium.Element(f'<h4 style="position:fixed;top:8px;left:60px;z-index:9999;background:white;'
                                               f'padding:4px 10px;border-radius:4px;font:14px sans-serif">{title}</h4>'))
    # road network near the track
    ax_, ay_, bx_, by_ = arc_geometry(d)
    lines = shapely.linestrings(np.stack([np.stack([ax_, ay_], 1), np.stack([bx_, by_], 1)], 1))
    near = np.unique(STRtree(lines).query(shapely.points(tr.x.values, tr.y.values), predicate="dwithin", distance=150.0)[1])
    fg = folium.FeatureGroup(name=f"Road network ({len(near):,} arcs within 150 m)")
    for x in near:
        folium.PolyLine(_arc_line(d, x), color=COL["road"], weight=2, opacity=.6).add_to(fg)
    fg.add_to(m)
    # GPS observations
    g = folium.FeatureGroup(name=f"GPS observations ({len(tr):,} fixes)")
    step = max(1, len(tr) // 3000)
    for k in range(0, len(tr), step):
        folium.CircleMarker((tr.lat.values[k], tr.lon.values[k]), radius=2.5, color=COL["gps"], weight=1, fill=True,
                            fill_opacity=.7, tooltip=f"fix {k}, t={tr.t.values[k]:.0f}s").add_to(g)
    folium.Marker((tr.lat.values[0], tr.lon.values[0]), tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(g)
    folium.Marker((tr.lat.values[-1], tr.lon.values[-1]), tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(g)
    g.add_to(m)
    return m


def eda_map(part, trip):
    d = load_trip(os.path.join(ROOT, "data_raw", part), trip, True)
    m = base_map(d, f"EDA - {part} trip {trip}: raw GPS vs road network vs reference route")
    if d["route"] is not None:
        _route_layer(d, d["route"], "Reference route (.route)", COL["ref"], 5).add_to(m)
    m.get_root().html.add_child(_legend([("Road network", COL["road"], 3, "0"), ("GPS fix", COL["gps"], 10, "50%"),
                                         ("Reference route", COL["ref"], 5, "0")]))
    folium.LayerControl(collapsed=False).add_to(m)
    os.makedirs(OUT, exist_ok=True); p = os.path.join(OUT, f"eda_{part}_{trip}.html"); m.save(p); return p


def result_map(part, trip, model=None):
    model = model or PathSnapModel(ROOT); pdir = os.path.join(ROOT, "data_raw", part)
    pred = model.predict_trip(pdir, trip); met = score_trip(pdir, trip, pred, cand=pred["cand"]) or {}
    d = load_trip(pdir, trip, True)
    title = (f"{part} trip {trip}: Top-1 classifier {100 * met.get('top1_classifier', 0):.1f}% -> HMM {100 * met.get('top1_hmm', 0):.1f}%"
             f" | route F1 {100 * met.get('route_f1_hmm', 0):.1f}%")
    m = base_map(d, title)
    if d["route"] is not None:
        _route_layer(d, d["route"], "Reference route", COL["ref"], 8).add_to(m)
    _route_layer(d, pred["pred_classifier"], "Classifier-only route", COL["cls"], 4, dash="6 6", show=False).add_to(m)
    _route_layer(d, pred["pred_hmm"], "Predicted route (HMM/Viterbi)", COL["hmm"], 4).add_to(m)
    m.get_root().html.add_child(_legend([("Road network", COL["road"], 3, "0"), ("GPS fix", COL["gps"], 10, "50%"),
                                         ("Reference route", COL["ref"], 6, "0"), ("Classifier-only route", COL["cls"], 4, "0"),
                                         ("Predicted route (HMM)", COL["hmm"], 4, "0")]))
    folium.LayerControl(collapsed=False).add_to(m)
    os.makedirs(OUT, exist_ok=True); p = os.path.join(OUT, f"result_{part}_{trip}.html"); m.save(p)
    return p, met


if __name__ == "__main__":
    if len(sys.argv) == 3:
        print(result_map(sys.argv[1], sys.argv[2])[0])
    else:
        print(eda_map("train", "00000001")); print(eda_map("student_test", "00000007"))
        mdl = PathSnapModel(ROOT)
        for t in ("00000076", "00000077", "00000007"):
            p, met = result_map("student_test", t, mdl)
            print(p, f"Top-1 {met['top1_classifier']:.4f} -> {met['top1_hmm']:.4f}")
