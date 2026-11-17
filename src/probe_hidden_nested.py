"""
probe_hidden_nested.py  (SERVER-SIDE)  —  unbiased AUROC via nested CV
=====================================================================
Addresses two points at once:

  (a) "best-performing depth" optimism bias: the original probe_hidden.py picked
      the layer that maximised CV performance and then reported that same
      performance. That inflates the estimate. Here we use NESTED cross-
      validation: the layer (depth fraction) is chosen on INNER folds only, and
      performance is measured on the held-out OUTER fold, which never influenced
      the choice. This gives an unbiased estimate.

  (b) missing AUROC: we report AUROC (threshold-free, imbalance-robust), the same
      metric used for the behavioural baselines, so the internal probe and the
      behavioural signals can be compared on equal footing.

Reads the saved 2-way hidden states hidden_<model>.npz (parametric vs
confabulated) produced by extract_hidden.py. No re-extraction needed.

Outer loop: 5 folds. For each outer fold, inner 4-fold CV over the remaining data
picks the depth fraction with the best inner AUROC; a probe is refit on all
outer-train data at that fraction and scored on the outer-test fold. We aggregate
outer-test predictions across folds and compute one AUROC per model.

OUTPUT: outputs/probe_hidden_nested.json
Usage:
    python probe_hidden_nested.py --model all --verbose
"""
from __future__ import annotations
import argparse, os, glob, json
import numpy as np


def _auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sp = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (sp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def _fit_score(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def _balance_idx(y, seed=0):
    y = np.asarray(y)
    i0 = np.where(y == 0)[0]; i1 = np.where(y == 1)[0]
    m = min(len(i0), len(i1))
    if m < 25:
        return None
    rng = np.random.RandomState(seed)
    return np.concatenate([rng.choice(i0, m, False), rng.choice(i1, m, False)])


def nested_auroc(fracs, Xby, y, outer_k=5, inner_k=4, seed=0):
    """Xby: dict frac->X (same row order); y: labels. Returns (auroc, chosen_fracs)."""
    n = len(y)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    outer = np.array_split(idx, outer_k)
    oof_pred = np.full(n, np.nan)
    chosen = []
    for oi in range(outer_k):
        te = outer[oi]
        tr = np.concatenate([outer[j] for j in range(outer_k) if j != oi])
        # inner CV to pick best frac (by inner AUROC)
        inner = np.array_split(rng.permutation(tr), inner_k)
        best_f, best_a = None, -1
        for f in fracs:
            X = Xby[f]
            inner_pred = np.full(len(tr), np.nan)
            pos = {v: k for k, v in enumerate(tr)}
            for ii in range(inner_k):
                ite = inner[ii]
                itr = np.concatenate([inner[j] for j in range(inner_k) if j != ii])
                p = _fit_score(X[itr], y[itr], X[ite])
                for local, gi in enumerate(ite):
                    inner_pred[pos[gi]] = p[local]
            a = _auroc(list(inner_pred), list(y[tr]))
            if a > best_a:
                best_a, best_f = a, f
        chosen.append(best_f)
        # refit on all outer-train at best_f, predict outer-test
        Xb = Xby[best_f]
        pred = _fit_score(Xb[tr], y[tr], Xb[te])
        oof_pred[te] = pred
    auroc = _auroc(list(oof_pred), list(y))
    return auroc, chosen


def run_model(path, verbose=False):
    d = np.load(path)
    fracs = [float(f) for f in d["fracs"]]
    y_all = d["y"].astype(int)
    key = os.path.basename(path).replace("hidden_", "").replace(".npz", "")

    # balance once (common row subset across all fracs), drop non-finite
    finite = np.ones(len(y_all), dtype=bool)
    for f in fracs:
        finite &= np.isfinite(d[f"X_{f}"].astype(np.float32)).all(axis=1)
    sel = _balance_idx(y_all[finite])
    if sel is None:
        print(f"  [skip] {key}: too few per class")
        return None
    globidx = np.where(finite)[0][sel]
    Xby = {f: d[f"X_{f}"].astype(np.float32)[globidx] for f in fracs}
    y = y_all[globidx]

    auroc, chosen = nested_auroc(fracs, Xby, y)
    from collections import Counter
    res = {"model": key, "nested_auroc": round(auroc, 4),
           "n_per_class": int((y == 0).sum()),
           "chosen_fracs": dict(Counter(chosen))}
    if verbose:
        print(f"  {key}: nested AUROC={auroc:.3f} "
              f"(n_per_class={res['n_per_class']}, chosen={res['chosen_fracs']})")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all")
    p.add_argument("--glob", default="outputs/hidden_*.npz")
    p.add_argument("--out", default="outputs/probe_hidden_nested.json")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    paths = sorted(glob.glob(args.glob)) if args.model == "all" \
        else [f"outputs/hidden_{args.model}.npz"]

    rows = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[skip] {path} not found"); continue
        print(f"[nested] {os.path.basename(path)}")
        r = run_model(path, verbose=args.verbose)
        if r:
            rows.append(r)
    if not rows:
        print("\nNo results. Need hidden_<model>.npz from extract_hidden.py.")
        return
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)

    import statistics as st
    a = [r["nested_auroc"] for r in rows]
    print(f"\n{'model':18s} {'nested_AUROC':>12s} {'n/class':>8s}")
    print("-" * 42)
    for r in rows:
        print(f"{r['model']:18s} {r['nested_auroc']:>12.3f} {r['n_per_class']:>8d}")
    print("-" * 42)
    print(f"{'MEAN':18s} {st.mean(a):>12.3f}")
    print(f"\nThis unbiased AUROC is directly comparable to the behavioural "
          f"baselines (self-consistency, semantic entropy). Compare means:")
    print(f"  internal hidden-state (nested):  {st.mean(a):.3f}")
    print(f"  behavioural semantic entropy:    ~0.77 (from baseline_uncertainty.json)")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
