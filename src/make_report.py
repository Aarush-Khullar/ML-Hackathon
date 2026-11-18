"""Build the technical report PDF (report/PathSnap_Technical_Report.pdf).
Every number, table and figure is read from results/ and artifacts/ (so the PDF is verifiable from the repo);
the prose comes from report/report_text.py, which the team must rewrite in its own words.
Usage: python src/make_report.py"""
import os, sys, json, importlib.util, datetime
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, Preformatted, KeepTogether)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, *p)
spec = importlib.util.spec_from_file_location("report_text", R("report", "report_text.py")); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)

ss = getSampleStyleSheet()
H1, H2, H3 = ss["Heading1"], ss["Heading2"], ss["Heading3"]
B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
SM = ParagraphStyle("sm", parent=B, fontSize=7.5, leading=9.5)
CAP = ParagraphStyle("cap", parent=B, fontSize=8, textColor=colors.HexColor("#555555"), spaceAfter=8)
story = []
P = lambda t, s=B: story.append(Paragraph(t, s))
pct = lambda x: "N/A" if pd.isna(x) else f"{100 * x:.2f}%"
f4 = lambda x: "N/A" if pd.isna(x) else f"{x:.4f}"


def table(rows, widths=None, font=7.5, header=True):
    data = [[Paragraph(str(c), SM) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#999999")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("FONTSIZE", (0, 0), (-1, -1), font), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header: st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8ecf2")))
    t.setStyle(TableStyle(st)); story.append(t); story.append(Spacer(1, 6))


def fig(path, caption, w=16.5):
    if not os.path.exists(path):
        P(f"<i>[missing figure: {os.path.relpath(path, ROOT)}]</i>"); return
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize(); h = w * ih / iw
    if h > 20: w, h = w * 20 / h, 20
    story.append(KeepTogether([Image(path, width=w * cm, height=h * cm), Paragraph(caption + f"  <font size=6>({os.path.relpath(path, ROOT)})</font>", CAP)]))


def on_page(c, doc):
    c.saveState(); c.setFont("Helvetica", 7); c.setFillColor(colors.grey)
    c.drawString(1.5 * cm, 1 * cm, "PathSnap technical report"); c.drawRightString(19.5 * cm, 1 * cm, f"page {doc.page}")
    if T.DRAFT:
        c.setFillColor(colors.HexColor("#d92d20")); c.setFont("Helvetica-Bold", 8)
        c.drawCentredString(10.5 * cm, 28.6 * cm, "DRAFT - narrative written with AI assistance; team must review and rewrite (report/report_text.py)")
    c.restoreState()


# ------------------------------------------------------------------ data
cmp_ = pd.read_csv(R("results", "models", "model_comparison_full.csv"))
qs = pd.read_csv(R("results", "preprocessing", "quality_summary.csv"))
ps = pd.read_csv(R("results", "eda", "partition_summary.csv"))
conf = json.load(open(R("artifacts", "model_config.json")))
fin = R("results", "models", "final_results_all_models.csv"); fin = pd.read_csv(fin) if os.path.exists(fin) else None
bp = R("results", "tuning", "best_params.json"); bp = json.load(open(bp)) if os.path.exists(bp) else None
tvu = R("results", "tuning", "tuned_vs_untuned.csv"); tvu = pd.read_csv(tvu) if os.path.exists(tvu) else None
hs = R("results", "hmm", "hmm_summary.csv"); hs = pd.read_csv(hs) if os.path.exists(hs) else None
hpt = R("results", "hmm", "hmm_comparison_per_trip.csv"); hpt = pd.read_csv(hpt, dtype={"trip": str}) if os.path.exists(hpt) else None
dcfg = json.load(open(R("artifacts", "decode_config.json")))

# ------------------------------------------------------------------ title
P("PathSnap: Distributed Map-Matching &amp; Route Reconstruction", ss["Title"])
P("UE24CS352A Machine Learning - Technical Report", ParagraphStyle("st", parent=B, alignment=1, fontSize=12))
story.append(Spacer(1, 10))
P(f"<b>Team:</b> {T.TEAM}<br/><b>Repository:</b> {T.REPO_URL}<br/><b>Generated:</b> {datetime.date.today()} by src/make_report.py "
  "from the result files in the repository.")
story.append(Spacer(1, 8)); P("1. Problem formulation", H1); P(T.INTRO)

# ------------------------------------------------------------------ 2.2 EDA
P("2. Dataset understanding and exploratory data analysis", H1)
P("Partitions used: EDA, preprocessing statistics and feature design use the 80 <b>train</b> and 10 <b>student_test</b> trips. "
  "The 10 evaluator trips were not used for any analysis or design decision.")
P("2.1 How we preprocessed the data", H2)
for x in T.PREPROCESSING: P("&bull; " + x)
P("2.2 Missing-value and data-quality analysis (train + student_test)", H2)
cols = ["partition", "trips", "gps_total", "nan", "nonmono", "dt_gt5", "step_gt100", "multi_comp_trips", "route_disconnected_pairs"]
table([["Partition", "Trips", "GPS fixes", "Missing values", "Non-monotonic t", "Gaps > 5 s", "Jumps > 100 m", "Trips with >1 network component", "Disconnected route pairs"]]
      + ps[cols].values.tolist())
fig(R("results", "eda", "fig_missing_quality.png"), "Figure 1. Data-quality checks per partition.", 13)
fig(R("results", "eda", "fig_dt_step.png"), "Figure 2. Sampling interval and step length between consecutive fixes (train + student_test).")
P("2.3 Patterns observed", H2)
for x in T.PATTERNS: P("&bull; " + x)
fig(R("results", "eda", "fig_trajectories.png"), "Figure 3. GPS trajectories (blue), reference route (green) and road network (grey) for three training trips.")
fig(R("results", "eda", "fig_road_network.png"), "Figure 4. Road-network connectivity: components per trip, largest-component share, node degrees.")
fig(R("results", "eda", "fig_trip_sizes.png"), "Figure 5. Trip sizes, extents and network sizes.")
fig(R("results", "preprocessing", "fig_preprocessing.png"), "Figure 6. Kinematic features: raw vs smoothed speed and heading, time gaps, speed, turning angle, candidate distances.")
fig(R("results", "maps", "eda_train_00000001.png"), "Figure 7. Folium map (train trip 00000001): raw GPS fixes, road network within 150 m and reference route, with layer control.")
P("2.4 Features and why we chose them", H2)
table([["Group", "Features", "Why"]] + [[g, f, w] for g, (f, w) in T.FEATURE_REASONS.items()], widths=[3.3 * cm, 5.5 * cm, 8.2 * cm])
fi = R("results", "models", "final_feature_importance.csv")
if os.path.exists(fi):
    d = pd.read_csv(fi).head(10)
    P("Top-10 features of the final LightGBM by gain share: " + ", ".join(f"{a} ({100 * b:.1f}%)" for a, b in zip(d.feature, d.gain_share)) + ".")

# ------------------------------------------------------------------ 3 models
story.append(PageBreak()); P("3. Model experimentation and comparison", H1)
P("Protocol: 8 models from 6 families trained on the 80 training trips (fitted on every 3rd fix, scored on all fixes), "
  "5-fold GroupKFold by trip for cross-validation, then refitted on all training trips and evaluated on the 10 student-test "
  "trips (19,160 fixes). Seed 42. Logs: results/models/training_log.txt.")
P("3.1 Overall model comparison", H2)
rows = [["Model", "Algorithm family", "Key configuration", "CV macro-F1 (±std)", "Top-1 accuracy (student)", "Route continuity", "Student-test macro-F1", "Log loss", "Inference time (ms / 1000 fixes)"]]
for _, r in cmp_.iterrows():
    rows.append([r.model, r.family, r.config, f"{r.cv_macro_f1:.4f} ± {r.cv_macro_f1_std:.4f}", pct(r.test_top1), pct(r.test_continuity),
                 f4(r.test_macro_f1), f4(r.test_log_loss), f"{1000 * r.inference_s_per_1000_fixes:.1f}"])
table(rows, widths=[2.2 * cm, 2 * cm, 3.6 * cm, 1.9 * cm, 1.6 * cm, 1.5 * cm, 1.5 * cm, 1.3 * cm, 1.6 * cm])
P("Top-1 accuracy is observation-level (argmax candidate vs derived reference arc). Route continuity = share of consecutive "
  "fixes whose predicted arcs are identical or connected within two arcs. All metrics are applicable to all models because "
  "every model outputs candidate probabilities.")
P(T.SELECTION)
P("3.2 Per-model analysis", H2)
for _, r in cmp_.iterrows():
    t = T.MODEL_TEXT.get(r.model, {})
    P(f"3.2.{_ + 1} {r.model} ({r.family})", H3)
    gap = r.train_macro_f1 - r.cv_macro_f1
    rows = [["Question", "Answer"],
            ["A1. Why suitable?", t.get("why", "")], ["A2. Strengths / limitations", f"{t.get('strengths', '')} Limitations: {t.get('limits', '')}"],
            ["B5. Hyperparameters (initial values)", r.config], ["B6. Loss / objective", t.get("loss", "")],
            ["C7. Student-test performance", f"accuracy {pct(r.test_accuracy)}, precision {pct(r.test_precision)}, recall {pct(r.test_recall)}, "
                                             f"F1 (on-road class) {pct(r.test_f1_pos)}, Top-1 after HMM {pct(r.test_top1_viterbi)}"],
            ["C8. Top-1 / macro-F1 / log loss", f"{pct(r.test_top1)} / {pct(r.test_macro_f1)} / {f4(r.test_log_loss)}"],
            ["C9. Over/underfitting", f"train macro-F1 {r.train_macro_f1:.4f}, CV {r.cv_macro_f1:.4f} (gap {gap:+.4f}), student-test "
                                      f"{r.test_macro_f1:.4f}. {t.get('fit', '')}"]]
    table(rows, widths=[3.6 * cm, 13.4 * cm], header=True)
P("3.3 Model visualisations", H2)
for f_, c_ in [("fig_cv_f1.png", "Figure 8. Cross-validation macro-F1 (mean ± std over 5 trip-grouped folds)."),
               ("fig_test_top1_f1.png", "Figure 9. Student-test candidate accuracy, macro-F1, Top-1 and Top-1 after HMM."),
               ("fig_logloss.png", "Figure 10. Student-test log loss (log scale)."),
               ("fig_overfit_gap.png", "Figure 11. Train vs CV macro-F1 (over/underfitting)."),
               ("fig_confusion_matrices.png", "Figure 12. Student-test confusion matrices (threshold 0.5)."),
               ("fig_mlp_loss.png", "Figure 13. MLP training and validation loss per epoch (only model with an epoch-wise history; "
                                    "validation = 8 held-out training trips)."),
               ("fig_feature_importance.png", "Figure 14. Final LightGBM feature importance (gain).")]:
    fig(R("results", "figures", f_), c_, 15)

# ------------------------------------------------------------------ 4 HPO
story.append(PageBreak()); P("4. Bayesian hyperparameter optimisation (LightGBM)", H1)
if bp:
    P(f"<b>Search:</b> Optuna {bp['sampler']} sampler with {bp['pruner']}, {bp['n_trials']} trials, {bp['search_time_s'] / 60:.1f} min. "
      f"<b>Objective:</b> {bp['objective']}. Trial 0 was the untuned default configuration.")
    P(T.HPO["why_space"])
    table([["Hyperparameter", "Range", "Scale", "Best value"]] +
          [[k, f"{v[0]} - {v[1]}", "log" if v[2] else "linear", f"{bp['best_params'][k]:.4g}" if isinstance(bp['best_params'][k], float) else bp['best_params'][k]]
           for k, v in bp["space"].items()], widths=[4 * cm, 4 * cm, 2 * cm, 4 * cm])
if tvu is not None:
    P("Tuned vs untuned (same training trips, same student-test trips):", B)
    table([["Model", "CV macro-F1", "Student Top-1", "Student macro-F1", "Student log loss", "Train time (s)", "Inference (ms / 1000 fixes)"]] +
          [[r.model, f4(r.cv_macro_f1), pct(r.student_top1), pct(r.student_macro_f1), f4(r.student_log_loss), f"{r.train_time_s:.1f}",
            f"{r.inference_ms_per_1000_fixes:.1f}"] for _, r in tvu.iterrows()])
fig(R("results", "tuning", "fig_optuna_history.png"), "Figure 15. Optuna optimisation history (CV macro-F1 per trial, best so far).", 13)
fig(R("results", "tuning", "fig_optuna_importance.png"), "Figure 16. Hyperparameter importance estimated by Optuna (fANOVA).", 12)

# ------------------------------------------------------------------ 5 HMM
story.append(PageBreak()); P("5. HMM and Viterbi route reconstruction", H1)
for q, k in [("1. Hidden states", "states"), ("2. Emission probabilities", "emission"), ("3. Transition probabilities", "transition"),
             ("4. Missing candidates and GPS gaps", "gaps"), ("5. Disconnected road segments", "disconnected"), ("Viterbi recurrence", "recurrence")]:
    P(f"<b>{q}.</b> {T.HMM[k]}")
P("Tuned decoder parameters (artifacts/decode_config.json): " + ", ".join(f"{k} = {v}" for k, v in dcfg.items() if not isinstance(v, str)))
P("5.1 Did HMM decoding improve the route? (student-test, final classifier)", H2)
if hs is not None:
    table([["Method", "Fixes", "Top-1", "Continuity", "Route breaks", "Route precision", "Route recall", "Route F1", "LCS ratio"]] +
          [[r.method, int(r.fixes), pct(r.top1), pct(r.continuity), int(r.breaks), pct(r.route_precision), pct(r.route_recall), pct(r.route_f1), pct(r.route_lcs_ratio)]
           for _, r in hs.iterrows()])
    P("Route precision/recall compare the set of arcs in the predicted route with the reference .route; the LCS ratio "
      "is the longest common subsequence of the two arc sequences divided by the reference length (order-aware).")
if hpt is not None:
    pv = hpt.pivot(index="trip", columns="method", values="top1")
    table([["Trip", "Classifier only", "HMM (no gap handling)", "HMM (full)", "GPS gaps > 5 s"]] +
          [[t, pct(r.get("classifier_only")), pct(r.get("hmm_no_gap")), pct(r.get("hmm_full")),
            int(hpt[(hpt.trip == t) & (hpt.method == "hmm_full")].gaps.iloc[0])] for t, r in pv.iterrows()])
P("<b>Where HMM did not help.</b> " + T.HMM["not_helped"])
fig(R("results", "hmm", "fig_hmm_per_trip.png"), "Figure 17. Top-1 per student-test trip: classifier-only vs HMM.", 15)
fig(R("results", "hmm", "fig_hmm_summary.png"), "Figure 18. Pointwise and sequence-level metrics.", 13)

# ------------------------------------------------------------------ 6 Folium
story.append(PageBreak()); P("6. Geographical visualisation using Folium", H1)
P("Maps are generated by src/folium_maps.py from the actual pipeline output; the interactive HTML maps are produced locally by that script (python src/folium_maps.py) and the PNG screenshots below are committed in results/maps/. "
  "Layers: road network within 150 m (grey), GPS fixes (blue dots, start/end markers), reference route (green), "
  "classifier-only route (orange dashed, off by default), HMM predicted route (red); legend and layer control on every map.")
for t_, c_ in [("00000076", "good case"), ("00000077", "large HMM gain"), ("00000007", "case where HMM did not help (vehicle parks at the end)")]:
    fig(R("results", "maps", f"result_student_test_{t_}.png"), f"Figure. Student-test trip {t_} ({c_}).", 16)

# ------------------------------------------------------------------ 7 API/UI
story.append(PageBreak()); P("7. API, UI and deployment", H1)
P(T.API)
P("Two-laptop test: laptop 1 (API) at 192.168.137.104:8000 and laptop 2 (UI) at 192.168.137.25, both on the same Wi-Fi "
  "hotspot. Laptop 2 runs <font face='Courier'>python ui/serve_ui.py --api http://192.168.137.104:8000</font>.")
fig(R("results", "deployment", "api_cmd_laptop1.png"),
    "Figure. API responses in the terminal of the receiving laptop (laptop 1): each POST /predict from laptop 2 "
    "(192.168.137.25) runs the full pipeline and logs the server-side Top-1 (classifier and HMM), route F1, continuity and run time.", 17)
fig(R("results", "deployment", "ui_laptop2.jpeg"),
    "Figure. Evaluator UI on laptop 2 after running evaluator trip 00000049: predicted HMM route on the map, metrics computed "
    "on the API server, and the history of trips run in this session.", 16)

# ------------------------------------------------------------------ final results
if fin is not None:
    P("8. Final results of all models on both held-out sets", H1)
    P("All models trained on the 80 training trips only. Evaluator-test numbers were computed after every model, "
      "hyperparameter and decoder choice was frozen; they were not used for any decision.")
    for part in ["student_test", "evaluator_test"]:
        d = fin[fin.test_set == part]
        table([["Model (" + part + ")", "Accuracy", "Macro-F1", "Precision", "Recall", "Log loss", "Top-1", "Top-1 + HMM"]] +
              [[r.model, pct(r.accuracy), pct(r.macro_f1), pct(r.precision), pct(r.recall), f4(r.log_loss), pct(r.top1), pct(r.top1_viterbi)]
               for _, r in d.iterrows()])
P("9. Conclusion", H1); P(T.CONCLUSION)
P("Reproducibility: every table and figure in this report is produced by the commands in README.md; AI assistance is disclosed in docs/AI_USE.md.")

os.makedirs(R("report"), exist_ok=True)
out = R("report", "PathSnap_Technical_Report.pdf")
SimpleDocTemplate(out, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                  title="PathSnap Technical Report").build(story, onFirstPage=on_page, onLaterPages=on_page)
print("report ->", out)
