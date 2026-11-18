"""Raw data loading and projection utilities for PathSnap."""
import os
import numpy as np
import pandas as pd
from pyproj import Transformer

# Trips come from different cities worldwide (Germany, Russia, USA, ...), so a single fixed
# CRS is wrong. Each trip is projected with its own Azimuthal Equidistant projection centred on
# the trip's GPS centroid: distances/bearings near the centre are metric-accurate (extent is < 50 km).

def make_projector(lon0, lat0):
    return Transformer.from_crs(
        "EPSG:4326", f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m", always_xy=True)


def list_trips(partition_dir):
    return sorted(d for d in os.listdir(partition_dir)
                  if os.path.isdir(os.path.join(partition_dir, d)))


def load_trip(partition_dir, trip_id, with_route=True):
    """Return dict with projected track, nodes, arcs (+ route if present)."""
    base = os.path.join(partition_dir, trip_id, trip_id)
    track = pd.read_csv(base + ".track", sep=r"\s+", header=None, names=["lon", "lat", "t"])
    nodes = pd.read_csv(base + ".nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(base + ".arcs", sep=r"\s+", header=None, names=["u", "v"])
    route = None
    rp = base + ".route"
    if with_route and os.path.exists(rp):
        route = pd.read_csv(rp, sep=r"\s+", header=None, names=["arc"])["arc"].to_numpy()
    lon0, lat0 = float(track["lon"].mean()), float(track["lat"].mean())
    T = make_projector(lon0, lat0)
    track["x"], track["y"] = T.transform(track["lon"].values, track["lat"].values)
    nodes["x"], nodes["y"] = T.transform(nodes["lon"].values, nodes["lat"].values)
    return dict(trip_id=trip_id, track=track, nodes=nodes, arcs=arcs, route=route, origin=(lon0, lat0))
