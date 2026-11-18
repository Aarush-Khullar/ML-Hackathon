"""Narrative text of the technical report.  >>> THE TEAM MUST REVIEW AND REWRITE THIS IN ITS OWN WORDS <<<

All tables, numbers and figures in the PDF are filled automatically from results/ by src/make_report.py.
Only the prose below is hand-written. It is a factual first draft written with AI assistance (see docs/AI_USE.md).
The event policy checks reports for AI-generated content and requires every member to be able to defend it, so
rewrite each paragraph in your own words, then set DRAFT = False and run:  python src/make_report.py
"""
DRAFT = True
TEAM = "Team name / members / SRNs  (fill in)"
REPO_URL = "https://github.com/<user>/<repo>  (fill in after pushing)"

INTRO = (
    "PathSnap matches noisy 1 Hz GPS trajectories to a road network. We cast the task as scoring candidate road arcs: "
    "for every GPS fix all directed arcs within 60 m (at most 12) are candidates, a classifier estimates the probability "
    "that the vehicle is on each candidate, and an HMM/Viterbi decoder combines these probabilities with road-network "
    "connectivity to output one arc per fix. Road ids are arbitrary per-trip integers in different cities, so they cannot "
    "be learned as classes directly; ranking candidates generalises across cities.")

PREPROCESSING = [
    "Loading: .track = lon, lat, t (no header); .nodes = lon, lat; .arcs = from_node, to_node (row index = arc id); "
    ".route = chronological arc ids. Column meanings were verified by plotting, not assumed.",
    "Projection: trips come from different countries, so each trip is projected with its own azimuthal-equidistant "
    "projection centred on its GPS centroid. All distances and bearings are then in metres instead of degrees.",
    "Validation: per trip we count missing values, out-of-range lat/lon, (0,0) points, duplicate rows and timestamps, "
    "non-monotonic timestamps, gaps > 5 s, implied speed > 50 m/s and jumps > 100 m.",
    "Cleaning: drop missing/impossible coordinates, sort by time (stable), drop duplicate fixes, keep the original index. "
    "Speed outliers and gaps are flagged rather than deleted, because the API must return a prediction for every fix.",
    "Candidate generation: STRtree query of all arcs within 60 m of each fix, nearest 12 kept; if none, the nearest arc "
    "within 300 m.",
    "Reference labels: the .route lists arcs but not which fix is on which arc, so each fix is aligned to a route arc "
    "with a monotone dynamic programme (minimise fix-to-arc distance, route position non-decreasing). Fixes whose "
    "aligned arc is more than 60 m away are excluded from training and cross-validation only.",
]

PATTERNS = [
    "The raw files are structurally clean: no missing values, duplicates or unordered timestamps in any partition.",
    "The real noise is spatial: GPS jitter of roughly 5-20 m, occasional jumps and long gaps (largest 42 h in train trip 90).",
    "Two-way roads are stored as two twin arcs with identical geometry, so distance alone cannot separate them; the "
    "direction of travel is needed.",
    "Per-step bearings at 1 Hz are erratic, especially at low speed, so heading must be smoothed over several fixes.",
    "Road networks are large (median about 19,500 arcs per training trip) and almost all have several disconnected components.",
    "Some reference routes cover only part of the track (e.g. train trips 33 and 54), which motivates the 60 m reliability filter.",
]

FEATURE_REASONS = {
    "Spatial (fix-arc)": ("dist, frac, proj_out, arc_len, dmin_end", "The true arc is usually the closest; position along the arc helps transitions."),
    "Direction": ("cos_hd, abs_ang, hd_valid", "Separates twin arcs of two-way roads; heading ignored when the vehicle barely moves."),
    "Long-range direction (v2)": ("cos_hd_w10, hd_valid_w10, cos_hd_w20, cos_hd_rank", "Direction stays visible when crawling or stopped; error analysis showed 4.3 % twin-arc errors in v1."),
    "Kinematic / progress": ("speed, ctx_prog, ctx_prog_w10, ctx_prog_w20", "Movement along vs against the arc over +-3/10/20 fixes."),
    "Ranking": ("dist_rank, dist_gap, dist_ratio, dist_z, n_cand", "Relative closeness among competing candidates matters more than absolute distance."),
    "Temporal context": ("ctx_dist_mean/max, ctx_dist_mean_w10/max_w10", "Smooths single noisy fixes using neighbouring fixes."),
    "Topology": ("out_deg_v, in_deg_u, has_twin, same_prev/next, conn_prev/next", "Is the arc connected to the candidates of neighbouring fixes?"),
}

MODEL_TEXT = {
    "M1_LogReg": dict(
        why="A linear, well-calibrated probabilistic baseline. It shows how much of the problem is linearly separable in our features.",
        strengths="Fast (about 2 s to train), produces probabilities directly, easy to interpret through its coefficients.",
        limits="Cannot model interactions such as distance gap x heading agreement, so it loses about 2 macro-F1 points to the tree ensembles.",
        loss="Binary cross-entropy (log loss) with L2 regularisation (C = 1), lbfgs solver, standardised inputs.",
        fit="Train macro-F1 is almost equal to CV and test, so it slightly underfits rather than overfits."),
    "M2_GaussianNB": dict(
        why="A simple generative probabilistic model, included to test whether class-conditional independent Gaussians are enough.",
        strengths="Trains in a fraction of a second; very high recall for the on-road class.",
        limits="Our features are strongly correlated (e.g. dist, dist_rank, dist_gap), which violates the independence assumption; "
               "probabilities are badly calibrated (log loss about 0.87) and precision is low.",
        loss="Maximum-likelihood estimates of per-class Gaussian means/variances (no iterative loss).",
        fit="Train, CV and test scores are all low and close together: clear underfitting / model bias."),
    "M3_DecisionTree": dict(
        why="A single non-linear tree can capture threshold rules such as 'closest candidate and heading agrees'.",
        strengths="Interpretable rules, no scaling needed, fast inference.",
        limits="High variance: train macro-F1 is clearly above CV, and its probabilities are coarse (highest log loss among trees).",
        loss="Gini impurity reduction at each split (max_depth 12, min_samples_leaf 20).",
        fit="The train-CV gap is the largest of all models, so it overfits somewhat."),
    "M4_RandomForest": dict(
        why="Bagging many decorrelated trees reduces the variance of the single tree while keeping non-linear rules.",
        strengths="Stable across folds, good probabilities, and the best Top-1 after HMM decoding on student-test.",
        limits="Larger model and slower inference than boosting; slightly lower classifier-only Top-1 than LightGBM.",
        loss="Gini impurity per tree; probabilities are averaged over 60 trees (bootstrap 50 %).",
        fit="Moderate train-CV gap; bagging keeps the test score close to CV."),
    "M5_HistGB": dict(
        why="Gradient boosting with histogram binning: sequentially corrects errors and handles large tabular data quickly.",
        strengths="Fast, strong accuracy, good calibration.",
        limits="Fewer tuning options than LightGBM/XGBoost; slightly below LightGBM on classifier-only metrics.",
        loss="Binary log loss minimised by gradient boosting (200 iterations, lr 0.1, 31 leaves).",
        fit="Small train-CV gap; generalises well."),
    "M6_LightGBM": dict(
        why="Leaf-wise gradient boosting on histograms; well suited to about 1.4 M tabular rows with interacting features.",
        strengths="Best CV macro-F1, best classifier-only Top-1 and lowest log loss (best HMM emissions); trains in seconds.",
        limits="More hyperparameters (leaves, learning rate, regularisation) and a larger train-CV gap than bagging; the "
               "differences to XGBoost/HistGB are within fold noise.",
        loss="Binary log loss minimised by gradient boosting (300 trees, lr 0.05, 63 leaves).",
        fit="Train macro-F1 is about 0.03 above CV, but CV and test agree, so the overfitting is mild."),
    "M7_XGBoost": dict(
        why="Regularised gradient boosting, a strong standard for tabular data, as a second boosting implementation.",
        strengths="Very close to LightGBM on every metric; fast with the hist tree method.",
        limits="Slightly lower Top-1 than LightGBM; depth-wise trees need more tuning of depth and regularisation.",
        loss="Binary log loss with L2 regularisation (300 trees, lr 0.1, max_depth 6).",
        fit="Similar train-CV gap to LightGBM; test agrees with CV."),
    "M8_MLP": dict(
        why="A shallow neural network learns non-linear feature combinations and produces probabilities.",
        strengths="Competitive accuracy with small inference cost; produces an epoch-wise learning curve.",
        limits="Needs feature scaling and is slower to train (about 50 s); slightly below the tree ensembles.",
        loss="Binary cross-entropy optimised with Adam (lr 1e-3, alpha 1e-4, batch 1024, 15 epochs).",
        fit="The train and validation loss curves stay close (see the MLP learning curve), so no strong overfitting."),
}

SELECTION = (
    "Primary selection metric: cross-validated macro-F1 on candidate pairs (5-fold GroupKFold by trip). Macro-F1 weights the "
    "minority 'on this arc' class equally, whereas plain accuracy is dominated by the roughly 88 % easy negatives. "
    "Observation-level Top-1, log loss (emission quality for the HMM) and inference time are used as tie-breakers. "
    "LightGBM has the best CV macro-F1, the best CV and student-test Top-1 and the lowest log loss, so it was selected. "
    "The four tree ensembles are within one fold standard deviation of each other, so this is a modest claim.")

HPO = dict(
    why_space="Ranges cover the regimes seen in the untuned comparison: number of trees and learning rate trade bias "
              "against variance; num_leaves and min_child_samples control tree complexity; subsample/colsample add "
              "randomness; L1/L2 penalties regularise leaf values. Log scales are used where the effect is multiplicative.",
    objective="Mean macro-F1 over 5-fold GroupKFold on the 80 training trips (the same metric used for model selection). "
              "Student-test and evaluator data are not used by the search.",
)

HMM = dict(
    states="For each GPS fix, the hidden states are that fix's candidate arcs (the same <=12 arcs within 60 m the classifier "
           "scored). Each state keeps its arc id, so the candidate-to-road-segment correspondence is preserved.",
    emission="alpha x log P(arc | fix), where P is the LightGBM probability (floored at 1e-4); alpha down-weights "
             "over-confident emissions.",
    transition="A cost table on the road graph: 0 for staying on the same arc, c_next for the successor arc, c_2hop for "
               "an arc two steps ahead, c_twin for a U-turn onto the twin arc and c_jump for unconnected arcs. The costs "
               "are tuned by coordinate search on out-of-fold predictions of the 80 training trips.",
    gaps="If the time step exceeds gap_s seconds, the successor test is replaced by a breadth-first search of up to max_hops "
         "arcs (cost c_next + c_hop x (hops - 1)). Fixes without any candidate are returned as unmatched (-1) and the chain "
         "continues from the previous fix.",
    disconnected="c_jump is finite, so when consecutive fixes only have unconnected candidates the decoder pays a large cost "
                 "and starts a new route piece instead of failing. These breaks are counted and reported.",
    recurrence="V_i(s) = em_i(s) + max_r [V_{i-1}(r) - cost(r -> s)], with back-pointers, then back-tracking from the "
               "best final state.",
    not_helped="HMM decoding hurts when the vehicle parks or crawls next to a two-way road (student trip 00000007 ends parked): "
               "the decoder keeps a consistent but wrong direction for a long stationary stretch, and the transition "
               "costs prevent it from switching.",
)

API = (
    "Laptop 1 runs the FastAPI server (api/main.py), which loads the saved LightGBM model, the feature configuration and the "
    "HMM parameters once at start-up. GET /trips lists the 10 reserved trips; POST /predict runs the complete pipeline "
    "(candidates, features, classifier, Viterbi) for the requested trip, then scores the prediction on the server against "
    "the reference route and returns only numeric metrics together with the GPS fixes and the predicted route geometry. "
    "Laptop 2 serves a static Leaflet page (ui/) that lists the trips, calls the API and draws the result. Every request "
    "is logged in the API terminal with client IP, trip, accuracy and run time.")

CONCLUSION = (
    "Long-range context features and HMM decoding were the two largest improvements. Hyperparameter tuning gave only a "
    "small gain. The remaining errors are mostly parked or crawling vehicles next to two-way roads and fix-to-arc "
    "boundaries at junctions, where the derived reference label itself is ambiguous within GPS noise.")
