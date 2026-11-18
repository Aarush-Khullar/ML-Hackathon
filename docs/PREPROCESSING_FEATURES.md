# PathSnap – Part 1: Data Preprocessing & Feature Engineering (1st evaluation cheat-sheet)

**Run live:** `python src/preprocess.py 00000007` (audit + cleaning + kinematic features) → `python src/preprocess_figs.py` (figure + example tables) → `python src/build_cache.py` (candidates + 32 model features, ~45 s).
Outputs: `results/preprocessing/` (quality_report.csv, quality_summary.csv, fig_preprocessing.png, example_*.csv).

## 1. Pipeline (in order)
| Step | What we do | Code |
|---|---|---|
| 0. Load | `.track` = lon, lat, t (no header); `.nodes` = lon, lat; `.arcs` = from, to; `.route` = arc ids (reference only). Verified column meaning by inspection, not assumed. | `data_utils.load_trip` |
| 1. Project | Trips are from **different cities worldwide** (Germany, Russia, USA, Kazakhstan…), so one fixed CRS is wrong. Each trip gets its own Azimuthal-Equidistant projection centred on its GPS centroid → coordinates in metres (degrees are not distances). | `make_projector` |
| 2. Validate | Count: missing values, lat/lon out of range, (0,0), duplicate rows, duplicate timestamps, non-monotonic time, gaps > 5 s, speed > 50 m/s, jumps > 100 m. | `preprocess.validate` |
| 3. Clean | Drop missing/impossible coordinates → sort by time (stable) → drop duplicate fixes; keep `orig_idx`. Implemented defensively; policy: fixes are never silently dropped at inference (API must answer per fix) – anomalies are **flagged**, not deleted. | `preprocess.clean` |
| 4. Kinematic features | step distance, dt, speed, acceleration, compass bearing, turning angle, cumulative distance, gap / stationary / speed-outlier flags. | `preprocess.kinematic_features` |
| 5. Candidate generation | For each fix, every directed road arc within 60 m (max 12 nearest; nearest arc ≤300 m if none). A fix–arc pair = one model row. | `features.build_trip` |
| 6. Road-related features | **32 features** per (fix, arc) – table below (23 in v1 + 9 long-range context features added in v2). | `features.py` |
| 7. Reference labels | `.route` lists arcs but not which fix is on which arc → per-fix label derived by monotone DP alignment (min. sum of point-to-arc distance, route position non-decreasing). Fixes whose aligned arc is >60 m away (route does not cover them) are excluded from **training/CV only**. A heading-aware variant (penalty when GPS heading opposes the arc) was tried and rejected: CV Top-1 0.906 → 0.904, student Top-1 0.922 → 0.914 (smoothed heading lags at turns). Kept as option `hd_pen`, default 0. | `features.align_labels` |

## 2. What the audit found (all 90 trips: 80 train + 10 student-test; 227,577 fixes)
| Check | train | student_test |
|---|---|---|
| Missing values / invalid lat-lon / (0,0) | 0 | 0 |
| Duplicate rows / duplicate timestamps | 0 | 0 |
| Non-monotonic timestamps | 0 | 0 |
| Time step exactly 1 s | 96.7 % (all trips) – ≤2 s: 98.6 % | |
| Gaps > 5 s | 1,216 (longest 153,064 s ≈ 42 h, trip 90) | 44 (longest 210 s) |
| Jumps > 100 m between consecutive fixes | 228 | 0 |
| Implied speed > 50 m/s (flagged, not removed) | 1,533 | 8 |
| Road network: trips with >1 disconnected component | 79 / 80 | 10 / 10 |
| Reference route not connected between consecutive arcs | 13 pairs | 1 pair |
| Fixes with unreliable derived label (excluded from training) | 22,345 of 208,417 (mostly trips 54 and 33 where the route covers only part of the track) | 25 of 19,160 kept in evaluation |

**Honest message:** the raw files are structurally clean (no NaN/duplicates/disorder); the real noise is *spatial* (GPS jitter of 5–20 m, jumps, gaps) and *reference-related* (route shorter than the track). That is why cleaning code is defensive but the heavy lifting is in features.

## 3. Features – what and why
| Group | Features | Why |
|---|---|---|
| Spatial (fix ↔ arc) | `dist`, `frac` (position along arc), `proj_out` (projection beyond arc ends), `arc_len`, `dmin_end` | Vehicle is usually closest to its true road; position along arc supports the HMM transition |
| **Direction** | `cos_hd`, `abs_ang`, `hd_valid` (smoothed GPS heading vs arc direction) | Two-way roads are two **twin arcs with identical geometry** – distance cannot separate them, direction can (example: dist 0.775 m for both, cos_hd +0.999 vs −0.999) |
| Kinematic | `speed`, `ctx_prog` (movement along the arc over ±3 fixes) | Moving along vs against the arc; heading is unreliable when stationary → `hd_valid` |
| Ranking | `dist_rank`, `dist_gap`, `dist_ratio`, `n_cand` | Relative closeness among competing candidates matters more than absolute distance (`dist_gap` was the most important v1 feature; see `results/models/final_feature_importance.csv` for v2) |
| Temporal context | `ctx_dist_mean`, `ctx_dist_max` (arc distance to fixes i±3) | Smooths single noisy fixes |
| **Long-range context (v2)** | `cos_hd_w10`, `hd_valid_w10`, `cos_hd_w20` (heading over ±10/±20 fixes, forward+backward filled), `ctx_dist_mean_w10`, `ctx_dist_max_w10` (arc distance to fixes ±5…±10), `ctx_prog_w10`, `ctx_prog_w20` (movement along arc over ±10/±20 fixes), `dist_z` (distance z-score among the fix's candidates), `cos_hd_rank` (heading-agreement rank among candidates) | Error analysis of v1 showed 4.3 % of fixes wrong on the **twin arc** (opposite direction, mostly at speed < 2 m/s where the ±3 heading is invalid) and 2.9 % on a side road sharing a node. A wider window still sees the direction of travel when the car is crawling or stopped. Effect: student Top-1 89.5 % → 92.2 % (same model family) |
| Topology / continuity | `out_deg_v`, `in_deg_u`, `has_twin`, `same_prev/next`, `conn_prev/next` | Is this arc connected to candidates of neighbouring fixes? Encodes road-network connectivity |

**Design decisions to justify:** (1) per-trip metric projection; (2) smoothing heading over ±3 fixes because per-step bearing is erratic at 1 Hz with ~10 m noise (see figure); (3) ignoring heading when displacement < 4 m (stationary); (4) flagging not deleting; (5) every preprocessing statistic uses only the current trip – no learned transform is fitted on test data, so there is **no leakage** (features are computed per trip; nothing is scaled globally except inside the Logistic-Regression/MLP pipelines, which are fitted on training rows only).

## 4. Likely viva questions
* *Why not Euclidean distance in degrees?* Degrees are angular; 1° lon ≠ 1° lat in metres and varies with latitude → projected to metres.
* *Why a local projection per trip?* The data span several continents; no single CRS is accurate everywhere.
* *How do you handle missing/duplicate/unordered timestamps?* `validate` counts them, `clean` removes/sorts them; none occur in this dataset (shown in quality_report.csv).
* *How do you handle large GPS gaps?* `gap_flag` (dt > 5 s); the HMM widens its route-search radius with the gap and uses a reset penalty when no connected route exists.
* *Why 60 m radius / 12 candidates?* Covers 99.8 % of student-test fixes (candidate miss rate 0.2 %) while keeping ~9 candidates per fix.
* *Is the label ground truth?* No – derived by DP alignment; the reference route is an assumption-laden reference (route shorter than track in some trips).
