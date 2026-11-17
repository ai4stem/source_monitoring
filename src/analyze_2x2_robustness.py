"""
analyze_2x2_robustness.py  —  checks on the matched correctness estimate (anchor support)
=====================================================================================
  seeds        matched AUROC over five seeds (matching order and CV folds), mean and SD
  nested       matched AUROC with the read-out depth chosen inside each training fold
               (inner 4-fold CV), removing the optimism of best-of-four depth selection
  mlp          one-hidden-layer MLP probe (256 units) at the best linear depth, same folds
  stability    can the hidden state read stability itself? unanimous vs non-unanimous
               facts, separately within correct and within incorrect facts (balanced)

CPU only.  OUTPUT  outputs/analysis_2x2_robustness.json
Usage:  python analyze_2x2_robustness.py --model all --jobs 13
"""
from __future__ import annotations
import argparse, glob, json, os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
from analyze_2x2_matched import caliper_match, _auroc, _cv_auroc
from analyze_2x2 import best_over_fracs


def _fit_predict(Xtr, ytr, Xte, kind="lin"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    if kind == "lin":
        clf = LogisticRegression(max_iter=1000)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, early_stopping=True,
                            max_iter=200, random_state=0)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def _folds(n, k, seed):
    idx = np.random.RandomState(seed).permutation(n)
    return np.array_split(idx, k)


def run_model(key, caliper=0.05):
    z = np.load(f"outputs/hidden2x2_{key}.npz"); s = np.load(f"outputs/support_semantic_{key}.npz")
    fracs = [float(f) for f in z["fracs"]]
    d = {f: z[f"X_{f}"].astype(np.float32) for f in fracs}
    correct = z["correct"].astype(int); support = s["support_anchor"].astype(float)
    ci = np.where(correct == 1)[0]; ii = np.where(correct == 0)[0]
    res = {"model": key}

    def matched_set(seed):
        mp, mn = caliper_match(support[ci], ci, support[ii], ii, caliper, seed=seed)
        return np.concatenate([mp, mn]), np.array([1] * len(mp) + [0] * len(mn))

    # ---- seeds
    vals = []
    for seed in range(5):
        gi, y = matched_set(seed)
        vals.append(max(_cv_auroc(d[f][gi], y, seed=seed) for f in fracs))
    res["seeds"] = {"values": [round(float(v), 4) for v in vals],
                    "mean": round(float(np.mean(vals)), 4), "sd": round(float(np.std(vals, ddof=1)), 4)}
    # ---- nested depth selection and MLP, seed 0 matched set
    gi, y = matched_set(0)
    per_depth = {f: _cv_auroc(d[f][gi], y, seed=0) for f in fracs}
    best_f = max(per_depth, key=per_depth.get)
    res["best_of_four"] = round(float(per_depth[best_f]), 4); res["best_depth"] = best_f
    res["per_depth"] = {str(f): round(float(v), 4) for f, v in per_depth.items()}
    folds = _folds(len(gi), 5, 0); oof = np.zeros(len(gi)); chosen = []
    for i in range(5):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(5) if j != i])
        inner = {f: _cv_auroc(d[f][gi][tr], y[tr], k=4, seed=1) for f in fracs}
        f_sel = max(inner, key=inner.get); chosen.append(f_sel)
        oof[te] = _fit_predict(d[f_sel][gi][tr], y[tr], d[f_sel][gi][te])
    res["nested"] = round(float(_auroc(list(oof), list(y))), 4); res["nested_depths"] = chosen
    oof = np.zeros(len(gi))
    for i in range(5):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(5) if j != i])
        oof[te] = _fit_predict(d[best_f][gi][tr], y[tr], d[best_f][gi][te], kind="mlp")
    res["mlp_at_best_depth"] = round(float(_auroc(list(oof), list(y))), 4)
    # ---- is stability itself legible?
    dd = {f"X_{f}": z[f"X_{f}"] for f in fracs}
    una = support >= 0.999
    a, _, n = best_over_fracs(dd, fracs, una & (correct == 1), ~una & (correct == 1))
    res["stability_within_correct"] = {"AUROC": round(float(a), 4), "n_per_class": int(n)}
    a, _, n = best_over_fracs(dd, fracs, una & (correct == 0), ~una & (correct == 0))
    res["stability_within_incorrect"] = {"AUROC": round(float(a), 4), "n_per_class": int(n)}
    print(f"[done] {key}: seeds {res['seeds']['mean']}±{res['seeds']['sd']} | best4 {res['best_of_four']} "
          f"nested {res['nested']} | mlp {res['mlp_at_best_depth']} | stab|corr {res['stability_within_correct']['AUROC']} "
          f"stab|inc {res['stability_within_incorrect']['AUROC']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all"); ap.add_argument("--jobs", type=int, default=13)
    ap.add_argument("--out", default="outputs/analysis_2x2_robustness.json")
    args = ap.parse_args()
    keys = sorted(os.path.basename(p)[10:-4] for p in glob.glob("outputs/hidden2x2_*.npz")) \
        if args.model == "all" else args.model.split(",")
    from multiprocessing import Pool
    with Pool(min(args.jobs, len(keys))) as pool:
        rows = pool.map(run_model, keys)
    json.dump(rows, open(args.out, "w"), indent=2)
    m = lambda f: float(np.mean([f(r) for r in rows]))
    print(f"\nmean: seeds {m(lambda r: r['seeds']['mean']):.3f} (mean SD {m(lambda r: r['seeds']['sd']):.3f}) | "
          f"best-of-four {m(lambda r: r['best_of_four']):.3f} | nested {m(lambda r: r['nested']):.3f} | "
          f"MLP {m(lambda r: r['mlp_at_best_depth']):.3f} | stability|correct {m(lambda r: r['stability_within_correct']['AUROC']):.3f} | "
          f"stability|incorrect {m(lambda r: r['stability_within_incorrect']['AUROC']):.3f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
