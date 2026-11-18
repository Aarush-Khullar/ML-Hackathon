"""PathSnap FastAPI backend  (LAPTOP 1: model + HMM + road network + server-side evaluation).

Run:   python api/main.py                 (listens on 0.0.0.0:8000 so the UI laptop can reach it)
  or   uvicorn api.main:app --host 0.0.0.0 --port 8000
Env:   PATHSNAP_TRIPS_DIR  folder with the 10 reserved evaluation trips (default data_raw/evaluator_test)

Endpoints
  GET  /health                 -> status, model and decoder config
  GET  /trips                  -> the evaluation trip ids (+ number of GPS fixes); no labels
  POST /predict  {"trip_id"}   -> full pipeline: features -> LightGBM -> HMM/Viterbi; returns GPS fixes, predicted
                                  arc per fix, predicted route geometry, candidate road geometry and server-side metrics.
The reference route is only used inside pipeline.score_trip(); it is never included in a response.
Every request is logged to this terminal (and to api/logs/requests.log).
"""
import os, sys, time, logging
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pipeline import PathSnapModel, score_trip, route_geometry  # noqa: E402
from data_utils import list_trips  # noqa: E402

TRIPS_DIR = os.environ.get("PATHSNAP_TRIPS_DIR", os.path.join(ROOT, "data_raw", "evaluator_test"))
os.makedirs(os.path.join(ROOT, "api", "logs"), exist_ok=True)
log = logging.getLogger("pathsnap")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(ROOT, "api", "logs", "requests.log"), encoding="utf-8")):
    h.setFormatter(_fmt); log.addHandler(h)

MODEL = PathSnapModel(ROOT)


@asynccontextmanager
async def lifespan(_app):
    log.info("=" * 70)
    log.info(f"PathSnap API ready | model {MODEL.conf['model']} | {len(MODEL.conf['features'])} features | HMM {MODEL.hmm_cfg}")
    log.info(f"evaluation trips from {TRIPS_DIR}: {list_trips(TRIPS_DIR)}")
    log.info("=" * 70)
    yield


app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan,
              description="Map-matching of noisy GPS trips: LightGBM candidate scoring + HMM/Viterbi")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return dict(status="ok", model=MODEL.conf["model"], n_features=len(MODEL.conf["features"]), hmm=MODEL.hmm_cfg)


@app.get("/trips")
def trips(request: Request):
    out = []
    for t in list_trips(TRIPS_DIR):
        with open(os.path.join(TRIPS_DIR, t, f"{t}.track")) as f:
            out.append(dict(trip_id=t, n_fixes=sum(1 for _ in f)))
    log.info(f"GET  /trips   from {request.client.host} -> {len(out)} trips")
    return out


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    t = req.trip_id.strip()
    if t not in list_trips(TRIPS_DIR):
        log.info(f"POST /predict from {request.client.host} trip={t} -> 404 unknown trip")
        raise HTTPException(404, f"unknown trip {t}")
    log.info(f"POST /predict from {request.client.host} trip={t}  ... running pipeline")
    t0 = time.time()
    pred = MODEL.predict_trip(TRIPS_DIR, t)
    metrics = score_trip(TRIPS_DIR, t, pred, cand=pred["cand"])          # server-side only
    d, cand = pred["trip"], pred["cand"]
    tr = d["track"]
    a = d["arcs"]; nodes = d["nodes"]
    road = []
    for x in np.unique(cand.arc.values):                                  # road segments near the trip (candidate arcs)
        u, v = int(a.u.values[x]), int(a.v.values[x])
        road.append([[float(nodes.lat.values[u]), float(nodes.lon.values[u])], [float(nodes.lat.values[v]), float(nodes.lon.values[v])]])
    resp = dict(
        trip_id=t, n_fixes=len(tr), n_candidates=int(len(cand)),
        gps=[[float(la), float(lo), float(tt)] for la, lo, tt in zip(tr.lat.values, tr.lon.values, tr.t.values)],
        predicted_arc_per_fix=[int(x) for x in pred["pred_hmm"]],
        predicted_route=route_geometry(d, pred["pred_hmm"]),
        classifier_route=route_geometry(d, pred["pred_classifier"]),
        road_network=road, metrics=metrics,
        timing={k: round(v, 3) for k, v in pred["timing"].items()} | {"request_s": round(time.time() - t0, 3)})
    m = metrics or {}
    log.info(f"     trip={t} fixes={len(tr)} candidates={len(cand)} | Top-1 classifier {100 * m.get('top1_classifier', float('nan')):.2f}% "
             f"-> HMM {100 * m.get('top1_hmm', float('nan')):.2f}% | route F1 {100 * m.get('route_f1_hmm', float('nan')):.2f}% "
             f"| continuity {100 * m.get('continuity_hmm', float('nan')):.2f}% | {time.time() - t0:.2f}s")
    return resp


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("PATHSNAP_HOST", "0.0.0.0"), port=int(os.environ.get("PATHSNAP_PORT", 8000)))
