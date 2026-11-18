"""Print the saved 8-model comparison (read from results/models/M*.json - no retraining).  python src/show_results.py"""
import os, json, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, "results", "models", "M?_*.json")))]
for r in R:
    c = r["test_confusion"]; r["acc"] = (c[0][0] + c[1][1]) / sum(map(sum, c)); r["name"] = r["model"].split("_", 1)[1]
R.sort(key=lambda r: -r["test_top1"])
line = "-" * 96
print("\nPathSnap - 8-model comparison   (Test = 10 student-test trips, CV = 5-fold trip-grouped on 80 train trips)")
print("\nTABLE 1: Test-set accuracy metrics (percent)\n" + line)
print(f"{'Rank':<5}{'Model':<14}{'Accuracy':>10}{'Top-1 Acc':>11}{'Macro-F1':>10}{'Precision':>11}{'Recall':>9}{'Log loss':>10}")
print(line)
for i, r in enumerate(R, 1):
    print(f"{i:<5}{r['name']:<14}{100*r['acc']:>10.2f}{100*r['test_top1']:>11.2f}{100*r['test_macro_f1']:>10.2f}"
          f"{100*r['test_precision']:>11.2f}{100*r['test_recall']:>9.2f}{r['test_log_loss']:>10.4f}")
print(line)
print("\nTABLE 2: Cross-validation, route continuity and speed\n" + line)
print(f"{'Rank':<5}{'Model':<14}{'CV macro-F1':>13}{'CV Top-1':>10}{'Continuity':>12}{'Train (s)':>11}{'Predict (s/1000 fixes)':>24}")
print(line)
for i, r in enumerate(R, 1):
    print(f"{i:<5}{r['name']:<14}{100*r['cv_macro_f1']:>13.2f}{100*r['cv_top1']:>10.2f}{100*r['test_continuity']:>12.2f}{r['train_time_s']:>11.1f}{r['inference_s_per_1000_fixes']:>24.4f}")
print(line)
print("Accuracy = all (GPS point, road) pairs classified correctly (looks high: most candidates are obviously wrong).")
print("Top-1 Acc = share of GPS points whose best-scored road is correct  <- main metric.   Ranked by Top-1.")
print(f"Selected model: {R[0]['name']}")
