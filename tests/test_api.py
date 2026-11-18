"""API contract tests.  Run: python -m pytest tests -q   (or: python tests/test_api.py)
Checks: trips are listed, every trip can be predicted, one arc per GPS fix is returned, unknown trips give 404,
and no reference-route geometry or per-fix reference labels are sent to the client."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "api"))
from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402

client = TestClient(main.app)
ALLOWED_KEYS = {"trip_id", "n_fixes", "n_candidates", "gps", "predicted_arc_per_fix", "predicted_route",
                "classifier_route", "road_network", "metrics", "timing"}


def test_trips_listed():
    r = client.get("/trips"); assert r.status_code == 200
    assert len(r.json()) >= 1 and all("trip_id" in t for t in r.json())


def test_unknown_trip_404():
    assert client.post("/predict", json={"trip_id": "does-not-exist"}).status_code == 404


def test_every_trip_predicts_without_leaking_labels():
    for t in client.get("/trips").json():
        j = client.post("/predict", json={"trip_id": t["trip_id"]}).json()
        assert set(j) == ALLOWED_KEYS                                  # no extra fields such as a reference route
        assert len(j["predicted_arc_per_fix"]) == len(j["gps"]) == t["n_fixes"]
        assert all(isinstance(v, (int, float, type(None))) for v in j["metrics"].values())   # metrics are numbers only
        assert 0.0 <= j["metrics"]["top1_hmm"] <= 1.0


if __name__ == "__main__":
    test_trips_listed(); test_unknown_trip_404(); test_every_trip_predicts_without_leaking_labels(); print("all API tests passed")
