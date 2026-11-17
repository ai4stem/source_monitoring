"""
analyze.py
==========
Evaluation metrics for the four groups we designed, plus result I/O.

  (A) source attribution quality : macro-F1, per-class P/R, confusion matrix
  (B) DRM confabulation rate     : critical-lure error rate (+ by BAS bucket)
  (C) conflict resolution        : verbatim-override rate vs over-deference rate
  (D) task preservation / OOD    : answer accuracy (hooks; computed upstream)

All metrics consume a list of per-claim prediction records (plain dicts), so
this module needs only numpy + sklearn -- runnable anywhere for analysis of
saved predictions.
"""

from __future__ import annotations
import json
import os
import numpy as np
from collections import defaultdict

SOURCE_CLASSES = ["verbatim", "parametric", "confabulated"]
IDX = {c: i for i, c in enumerate(SOURCE_CLASSES)}


# ---------------------------------------------------------------------------
# Prediction record format (one dict per evaluated claim):
#   {
#     "instance_id": str, "probe_kind": str, "item_type": str,
#     "true_source": "verbatim"|"parametric"|"confabulated"|None,
#     "pred_source": same set,
#     "list_type": "high_BAS"|... (drm only),
#     "answered_yes": bool        (drm probes: did model claim the word was present),
#     "followed_context": bool    (conflict probes: did model take the verbatim answer),
#     "task_correct": bool        (general QA accuracy),
#   }
# ---------------------------------------------------------------------------

# ---- (A) source attribution quality ----

def confusion_matrix(records) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=int)
    for r in records:
        ts, ps = r.get("true_source"), r.get("pred_source")
        if ts in IDX and ps in IDX:
            cm[IDX[ts], IDX[ps]] += 1
    return cm


def attribution_metrics(records) -> dict:
    cm = confusion_matrix(records)
    metrics = {"confusion_matrix": cm.tolist(),
               "classes": SOURCE_CLASSES, "per_class": {}}
    f1s = []
    for c, i in IDX.items():
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = int(cm[i, :].sum())
        metrics["per_class"][c] = {"precision": round(prec, 4),
                                   "recall": round(rec, 4), "f1": round(f1, 4),
                                   "support": support}
        if support > 0:                 # macro-F1 over present classes only
            f1s.append(f1)
    total = cm.sum()
    metrics["accuracy"] = round(np.trace(cm) / total, 4) if total else 0.0
    metrics["macro_f1"] = round(float(np.mean(f1s)), 4) if f1s else 0.0
    # confabulated recall is the safety-critical headline number
    cr = metrics["per_class"]["confabulated"]
    metrics["confabulated_recall"] = cr["recall"] if cr["support"] > 0 else None
    return metrics


# ---- (B) DRM confabulation rate ----

def drm_confabulation_rate(records) -> dict:
    """Critical-lure error rate; specificity vs studied/unrelated controls."""
    buckets = {"critical_lure": [], "studied": [], "unrelated": []}
    by_bas = defaultdict(list)
    for r in records:
        pk = r.get("probe_kind")
        if pk in buckets and "answered_yes" in r:
            buckets[pk].append(bool(r["answered_yes"]))
            if pk == "critical_lure":
                by_bas[r.get("list_type", "NA")].append(bool(r["answered_yes"]))

    def rate(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    out = {
        "critical_lure_false_yes_rate": rate(buckets["critical_lure"]),  # confab rate
        "studied_hit_rate": rate(buckets["studied"]),                    # should be high
        "unrelated_false_yes_rate": rate(buckets["unrelated"]),          # should be low
        "by_BAS": {k: rate(v) for k, v in sorted(by_bas.items())},
    }
    # specificity check: confab rate should exceed the unrelated baseline
    cl, ul = out["critical_lure_false_yes_rate"], out["unrelated_false_yes_rate"]
    out["drm_effect_present"] = (cl is not None and ul is not None and cl > ul)
    return out


# ---- (C) conflict resolution & over-deference ----

def conflict_metrics(records) -> dict:
    conflict, nonconflict = [], []
    for r in records:
        pk = r.get("probe_kind")
        if pk == "conflict" and "followed_context" in r:
            conflict.append(bool(r["followed_context"]))
        elif pk == "non_conflict_control" and "followed_context" in r:
            # for controls, followed_context=True means it (wrongly) deferred to
            # context instead of using parametric -> over-deference signal.
            nonconflict.append(bool(r["followed_context"]))

    def rate(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "verbatim_override_rate": rate(conflict),       # higher = healthier
        "over_deference_rate": rate(nonconflict),       # lower = healthier
    }


# ---- (D) task preservation ----

def task_accuracy(records) -> dict:
    xs = [bool(r["task_correct"]) for r in records if "task_correct" in r]
    return {"task_accuracy": round(sum(xs) / len(xs), 4) if xs else None,
            "n": len(xs)}


def evaluate_all(records) -> dict:
    return {
        "attribution": attribution_metrics(records),
        "drm": drm_confabulation_rate(records),
        "conflict": conflict_metrics(records),
        "task": task_accuracy(records),
    }


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def save_results(results: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_results(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_predictions(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_predictions(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out
