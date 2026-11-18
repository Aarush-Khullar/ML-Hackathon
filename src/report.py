"""Readable terminal report of one model's training/evaluation results (shared by run_experiments.py and show_results.py)."""
import sys

LINE = "=" * 78


def model_block(r):
    """Return a multi-line string summarising one model's results dict (as stored in results/models/<model>.json)."""
    out = [LINE, f"{r['model']}   [{r.get('family', '')}]", f"config : {r.get('config', '')}", "-" * 78,
           "5-fold GroupKFold CV on 80 training trips (no trip in two folds):"]
    folds = r.get("cv_fold_macro_f1")
    if folds:
        for k, (f1, t1, ll) in enumerate(zip(folds, r["cv_fold_top1"], r["cv_fold_logloss"]), 1):
            out.append(f"  fold {k}:  macro-F1 {f1:.4f}   Top-1 {t1:.4f}   log loss {ll:.4f}")
    out += [f"  MEAN  :  macro-F1 {r['cv_macro_f1']:.4f} +/- {r['cv_macro_f1_std']:.4f}   Top-1 {r['cv_top1']:.4f} +/- {r['cv_top1_std']:.4f}"
            f"   log loss {r['cv_logloss']:.4f}   ({r['cv_time_s']:.1f} s)",
            f"Final fit on all 80 training trips: {r['train_time_s']:.1f} s   train macro-F1 {r['train_macro_f1']:.4f}"
            f"   (gap to CV {r['train_macro_f1'] - r['cv_macro_f1']:+.4f})",
            "Student-test (10 unseen trips, 19,160 GPS fixes):",
            f"  candidate accuracy {100 * r['test_accuracy']:.2f} %   macro-F1 {100 * r['test_macro_f1']:.2f} %"
            f"   precision {100 * r['test_precision']:.2f} %   recall {100 * r['test_recall']:.2f} %",
            f"  Top-1 road accuracy {100 * r['test_top1']:.2f} %   Top-1 + Viterbi {100 * r['test_top1_viterbi']:.2f} %"
            f"   log loss {r['test_log_loss']:.4f}   continuity {100 * r['test_continuity']:.2f} %"]
    if "test_confusion" in r:
        (tn, fp), (fn, tp) = r["test_confusion"]
        out.append(f"  confusion: TN {tn:,}  FP {fp:,}  FN {fn:,}  TP {tp:,}")
    out.append(f"  inference {1000 * r['inference_s_per_1000_fixes']:.1f} ms per 1000 fixes")
    return "\n".join(out)


def summary_table(rows):
    hdr = f"{'model':17s}{'CV F1':>8s}{'CV Top1':>9s}{'Test acc':>10s}{'Test F1':>9s}{'Top-1':>8s}{'+Viterbi':>10s}{'logloss':>9s}{'train s':>9s}"
    out = [LINE, "SUMMARY (selection metric = CV macro-F1)", hdr, "-" * len(hdr)]
    best = max(rows, key=lambda r: r["cv_macro_f1"])["model"]
    for r in rows:
        out.append(f"{r['model']:17s}{r['cv_macro_f1']:8.4f}{r['cv_top1']:9.4f}{r['test_accuracy']:10.4f}{r['test_macro_f1']:9.4f}"
                   f"{r['test_top1']:8.4f}{r['test_top1_viterbi']:10.4f}{r['test_log_loss']:9.4f}{r['train_time_s']:9.1f}"
                   + ("   <- selected" if r["model"] == best and len(rows) > 1 else ""))
    return "\n".join(out)


class Tee:
    """Write everything printed to the terminal also to a log file."""
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8"); self.o = sys.stdout

    def write(self, s):
        self.o.write(s); self.f.write(s)

    def flush(self):
        self.o.flush(); self.f.flush()
