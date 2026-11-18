"""Thin compatibility wrapper around hmm.py (the single HMM/Viterbi implementation).
Older scripts call decode_trip(df, p, arcs, costs) / load_costs(root); see hmm.py for the model description."""
from hmm import decode_trip as _decode, load_config, successor_nodes, DEFAULTS  # noqa: F401

DEFAULT_COSTS = DEFAULTS


def load_costs(root):
    return load_config(root)


def decode_trip(df, p, arcs, costs, t=None, n_fix=None):
    return _decode(df, p, arcs, costs, t=t, n_fix=n_fix)
