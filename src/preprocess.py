"""Stage 1 - GPS preprocessing: validate -> clean -> kinematic features.   Usage: python src/preprocess.py [trip_id] [partition]
Policy: fixes are NEVER silently dropped by the model pipeline (the API must return one prediction per fix);
problems are detected, counted and either repaired (sort / de-duplicate / drop impossible coordinates, keeping `orig_idx`) or only flagged."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_trip, list_trips
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_SPEED = 50.0     # m/s (=180 km/h): faster implied movement is physically implausible for road vehicles -> flagged
GAP_S = 5.0          # time gap above which consecutive fixes are treated as a GPS gap
STILL = 0.5          # m/s: below this the fix is 'stationary' (heading unreliable)


def validate(tr):
    """Data-quality audit of a raw .track table (columns lon, lat, t, x, y)."""
    dt = np.diff(tr.t.values); step = np.hypot(np.diff(tr.x.values), np.diff(tr.y.values))
    sp = step / np.maximum(dt, 1e-3)
    return dict(n=len(tr), missing=int(tr[["lon", "lat", "t"]].isna().sum().sum()),
                bad_lat=int((tr.lat.abs() > 90).sum()), bad_lon=int((tr.lon.abs() > 180).sum()),
                zero_zero=int(((tr.lon == 0) & (tr.lat == 0)).sum()),
                dup_rows=int(tr.duplicated(["lon", "lat", "t"]).sum()), dup_timestamps=int(tr.t.duplicated().sum()),
                non_monotonic=int((dt <= 0).sum()), gaps_gt5s=int((dt > GAP_S).sum()), max_gap_s=float(dt.max()) if len(dt) else 0.0,
                irregular_dt_pct=round(100 * float(np.mean(dt != 1)), 2),
                speed_outliers=int((sp > MAX_SPEED).sum()), jumps_gt100m=int((step > 100).sum()),
                stationary_pct=round(100 * float(np.mean(sp < STILL)), 2))


def clean(tr):
    """Return cleaned track + orig_idx. Order: drop missing/impossible coordinates -> sort by time -> drop duplicate fixes."""
    tr = tr.copy(); tr["orig_idx"] = np.arange(len(tr))
    ok = tr[["lon", "lat", "t"]].notna().all(axis=1) & (tr.lat.abs() <= 90) & (tr.lon.abs() <= 180) & ~((tr.lon == 0) & (tr.lat == 0))
    tr = tr[ok]
    tr = tr.sort_values("t", kind="stable")
    tr = tr.drop_duplicates(["lon", "lat", "t"]).drop_duplicates("t", keep="first")
    return tr.reset_index(drop=True)


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def kinematic_features(tr):
    """Per-fix movement features from the projected (metric) coordinates."""
    f = tr.copy(); x, y, t = f.x.values, f.y.values, f.t.values
    dx, dy = np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0]); dt = np.diff(t, prepend=t[0] - 1.0)
    f["dt"] = dt; f["step_dist"] = np.hypot(dx, dy); f["speed"] = f.step_dist / np.maximum(dt, 1e-3)
    f["accel"] = np.diff(f.speed.values, prepend=f.speed.values[0]) / np.maximum(dt, 1e-3)
    brg = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0           # compass bearing: 0 = north, clockwise
    still = f.speed.values < STILL
    brg = pd.Series(np.where(still, np.nan, brg)).ffill().bfill().values  # stationary: keep last reliable heading
    f["bearing"] = brg; f["turn_angle"] = wrap180(np.diff(brg, prepend=brg[0]))
    f["cum_dist"] = np.cumsum(f.step_dist.values)
    f["gap_flag"] = (dt > GAP_S).astype(int); f["stationary_flag"] = still.astype(int)
    f["speed_outlier_flag"] = (f.speed.values > MAX_SPEED).astype(int)
    return f


if __name__ == "__main__":
    out = os.path.join(ROOT, "results", "preprocessing"); os.makedirs(out, exist_ok=True)
    rows = []
    for part in [x for x in ["train", "student_test", "evaluator_test"] if os.path.isdir(os.path.join(ROOT, "data_raw", x))]:
        pdir = os.path.join(ROOT, "data_raw", part)
        for tid in list_trips(pdir):
            raw = load_trip(pdir, tid, False)["track"]; q = validate(raw); c = clean(raw)
            q.update(trip=tid, partition=part, removed_by_cleaning=len(raw) - len(c)); rows.append(q)
    rep = pd.DataFrame(rows); rep.to_csv(os.path.join(out, "quality_report.csv"), index=False)
    agg = rep.groupby("partition")[["n", "missing", "bad_lat", "bad_lon", "zero_zero", "dup_rows", "dup_timestamps", "non_monotonic", "gaps_gt5s",
                                    "speed_outliers", "jumps_gt100m", "removed_by_cleaning"]].sum()
    agg["max_gap_s"] = rep.groupby("partition").max_gap_s.max(); print(agg.T.to_string()); agg.to_csv(os.path.join(out, "quality_summary.csv"))
    # demo on one trip
    tid = sys.argv[1] if len(sys.argv) > 1 else "00000007"; part = sys.argv[2] if len(sys.argv) > 2 else "student_test"
    raw = load_trip(os.path.join(ROOT, "data_raw", part), tid, False)["track"]; c = clean(raw); k = kinematic_features(c)
    print(f"\nTrip {tid}: raw {len(raw)} fixes -> cleaned {len(c)}"); print(json.dumps(validate(raw), indent=1))
    cols = ["t", "lon", "lat", "x", "y", "dt", "step_dist", "speed", "accel", "bearing", "turn_angle", "gap_flag", "stationary_flag"]
    k[cols].head(25).round(3).to_csv(os.path.join(out, f"example_kinematic_features_{tid}.csv"), index=False); print(k[cols].head(8).round(2).to_string())
