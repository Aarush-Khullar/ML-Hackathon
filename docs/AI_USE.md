# AI assistance disclosure (DRAFT – the team must review and complete this)

The event policy requires AI assistance to be disclosed. This draft lists what an AI coding assistant
(Claude Code, Anthropic) did in this repository. **Edit it so it accurately describes your team's own work**,
add which parts each team member did, and remove anything that is not true.

## What the AI assistant did
* Wrote and refactored Python code in `src/`, `api/`, `ui/` and `tests/` at the team's request, including the v2 long-range
  features, the HMM/Viterbi decoder with gap handling, the Optuna search script, the FastAPI backend, the Leaflet UI and
  the Folium map code.
* Ran the experiments locally on the team's laptop (feature cache, 8-model comparison, Bayesian optimisation, HMM
  experiments). All numbers in `results/` come from these real runs; nothing was typed in by hand.
* Performed error analysis (twin-arc and junction errors) that motivated the long-range features.
* Generated the figures, maps and a first draft of the documentation and report tables from the result files.

## What the team did
* _(fill in: problem understanding, design decisions you made or approved, checks you performed, which parts of
  the report you wrote yourselves, who is responsible for which component)_

## Checks
* Every reported metric can be regenerated with the commands in `README.md`.
* Each team member should be able to explain the preprocessing, features, model comparison, Bayesian optimisation,
  HMM, API and UI (see `docs/`).
