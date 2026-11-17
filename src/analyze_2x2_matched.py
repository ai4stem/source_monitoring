"""
analyze_2x2_matched.py  (SERVER-SIDE)  —  strict caliper-matched truth test
===========================================================================
The tercile version (analyze_2x2.py) left a residual support difference between
the truth classes in some models, so its support-only baseline rose above 0.5
(e.g. qwen3-8b 0.64, qwen3-14b 0.69), meaning support was not fully held fixed.
This script removes that leakage with 1:1 CALIPER MATCHING on the LOPO support
value: each incorrect item is paired with a correct item whose support is within
a small caliper, and unmatched items are dropped. On the matched set the two
truth classes have (by construction) almost identical support distributions, so
any remaining hidden-state separability is truth, not support.

We verify the match worked by reporting, on the matched set:
  * support-only AUROC for truth   (must be ~0.5)
  * entropy-only AUROC for truth    (must be ~0.5)
  * mean support gap                (must be ~0)
and then the hidden-state truth AUROC (nested-free, best depth, balanced) which
is the number of interest. We also give the hidden-state SUPPORT AUROC on a
truth-matched set (the mirror test) so Q1 and Q2 are both leakage-controlled.

OUTPUT: outputs/analysis_2x2_matched.json
Usage:
    python analyze_2x2_matched.py --model all --caliper 0.05 --verbose
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


def _cv_auroc(X, y, k=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or len(y) < 50:
        return float("nan")
    rng = np.random.RandomState(seed); idx = rng.permutation(len(X))
    folds = np.array_split(idx, k); oof = np.zeros(len(X))
    for i in range(k):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        if len(np.unique(y[tr])) < 2:
            oof[te] = 0.5; continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return _auroc(list(oof), list(y))


def caliper_match(val_pos, idx_pos, val_neg, idx_neg, caliper, seed=0):
    """Greedy 1:1 nearest-neighbour matching on a scalar (val) within caliper.
    Returns matched global indices (pos_matched, neg_matched)."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(idx_pos))
    used_neg = np.zeros(len(idx_neg), dtype=bool)
    mp, mn = [], []
    neg_sorted = np.argsort(val_neg)
    neg_vals_sorted = val_neg[neg_sorted]
    for oi in order:
        v = val_pos[oi]
        # find nearest unused neg within caliper via searchsorted
        lo = np.searchsorted(neg_vals_sorted, v - caliper, "left")
        hi = np.searchsorted(neg_vals_sorted, v + caliper, "right")
        best = -1; bestd = 1e9
        for k in range(lo, hi):
            j = neg_sorted[k]
            if used_neg[j]:
                continue
            dd = abs(val_neg[j] - v)
            if dd < bestd:
                bestd = dd; best = j
        if best >= 0:
            used_neg[best] = True
            mp.append(idx_pos[oi]); mn.append(idx_neg[best])
    return np.array(mp, dtype=int), np.array(mn, dtype=int)


def hidden_auroc_over_depth(d, fracs, gi, y, seed=0):
    best = -1
    for f in fracs:
        X = d[f"X_{f}"].astype(np.float32)[gi]
        fin = np.isfinite(X).all(axis=1)
        a = _cv_auroc(X[fin], np.asarray(y)[fin], seed=seed)
        if a == a and a > best:
            best = a
    return None if best < 0 else round(best, 4)


def run_model(path, caliper=0.05, verbose=False):
    d = np.load(path)
    fracs = [float(f) for f in d["fracs"]]
    support = d["support"].astype(float)
    correct = d["correct"].astype(int)
    entropy = d["entropy"].astype(float)
    key = os.path.basename(path).replace("hidden2x2_", "").replace(".npz", "")

    ci = np.where(correct == 1)[0]
    ii = np.where(correct == 0)[0]
    if len(ci) < 60 or len(ii) < 60:
        print(f"  [skip] {key}: too few in a truth class"); return None

    # ---- Q2 matched: match correct vs incorrect on SUPPORT ----
    mp, mn = caliper_match(support[ci], ci, support[ii], ii, caliper)
    if len(mp) < 60:
        print(f"  [skip] {key}: only {len(mp)} matched pairs at caliper {caliper}")
        return None
    gi = np.concatenate([mp, mn])
    y_truth = np.array([1] * len(mp) + [0] * len(mn))

    # leakage checks on matched set (must be ~0.5 / ~0)
    sup_only = round(_auroc(list(support[gi]), list(y_truth)), 4)
    ent_only = round(_auroc(list(-entropy[gi]), list(y_truth)), 4)
    sup_gap = round(float(support[mp].mean() - support[mn].mean()), 4)
    # the number of interest: hidden-state truth AUROC on support-matched set
    truth_auroc = hidden_auroc_over_depth(d, fracs, gi, y_truth)

    # ---- Q1 mirror: match high vs low support on TRUTH (control truth) ----
    # define support high/low by median, then match on correctness proportion is
    # trivial (binary), so instead match within-correct high vs low support by
    # entropy is not needed; we simply report support AUROC within correct only.
    med = np.median(support)
    hi_mask = support > med
    # within correct: high-support vs low-support, balanced
    c_hi = np.where((correct == 1) & hi_mask)[0]
    c_lo = np.where((correct == 1) & ~hi_mask)[0]
    q1 = None
    if len(c_hi) >= 60 and len(c_lo) >= 60:
        m = min(len(c_hi), len(c_lo))
        rng = np.random.RandomState(0)
        sel = np.concatenate([rng.choice(c_hi, m, False), rng.choice(c_lo, m, False)])
        y_sup = np.array([1] * m + [0] * m)
        q1 = hidden_auroc_over_depth(d, fracs, sel, y_sup)

    res = {"model": key, "n_matched_pairs": int(len(mp)),
           "caliper": caliper,
           "Q2_truth_matched_AUROC": truth_auroc,
           "Q2_support_only_AUROC": sup_only,
           "Q2_entropy_only_AUROC": ent_only,
           "Q2_support_gap": sup_gap,
           "Q1_support_within_correct_AUROC": q1}
    if verbose:
        print(f"  {key}: matched_pairs={len(mp)}")
        print(f"    Q2 truth|support-matched  AUROC={truth_auroc}  "
              f"(support-only={sup_only}, ent-only={ent_only}, gap={sup_gap})")
        print(f"    Q1 support|within-correct AUROC={q1}")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all")
    p.add_argument("--glob", default="outputs/hidden2x2_*.npz")
    p.add_argument("--out", default="outputs/analysis_2x2_matched.json")
    p.add_argument("--caliper", type=float, default=0.05)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    paths = sorted(glob.glob(args.glob)) if args.model == "all" \
        else [f"outputs/hidden2x2_{args.model}.npz"]

    rows = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[skip] {path} not found"); continue
        print(f"[matched] {os.path.basename(path)}")
        r = run_model(path, caliper=args.caliper, verbose=args.verbose)
        if r:
            rows.append(r)
    if not rows:
        print("\nNo results."); return
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)

    import statistics as st
    def m(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return st.mean(v) if v else float("nan")

    print(f"\n{'model':18s} {'pairs':>6s} {'Q2 truth':>9s} {'sup-only':>9s} "
          f"{'ent-only':>9s} {'Q1 sup':>8s}")
    print("-" * 62)
    for r in rows:
        print(f"{r['model']:18s} {r['n_matched_pairs']:>6d} "
              f"{str(r['Q2_truth_matched_AUROC']):>9s} "
              f"{str(r['Q2_support_only_AUROC']):>9s} "
              f"{str(r['Q2_entropy_only_AUROC']):>9s} "
              f"{str(r['Q1_support_within_correct_AUROC']):>8s}")
    print("-" * 62)
    print(f"{'MEAN':18s} {'':>6s} {m('Q2_truth_matched_AUROC'):>9.3f} "
          f"{m('Q2_support_only_AUROC'):>9.3f} {m('Q2_entropy_only_AUROC'):>9.3f} "
          f"{m('Q1_support_within_correct_AUROC'):>8.3f}")
    print(f"\nINTERPRETATION (strict, leakage-controlled):")
    print(f"  On the support-matched set, support-only and entropy-only AUROC should")
    print(f"  be ~0.5 (support truly held fixed). The hidden-state truth AUROC is then")
    print(f"  a clean estimate of how much TRUTH the representation carries independent")
    print(f"  of support. Compare Q2 (truth|matched) with Q1 (support|within-correct):")
    print(f"    Q2 > Q1  => truth is represented at least as strongly as support")
    print(f"    Q2 ~ 0.5 => truth vanishes once support is matched (support-only view)")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
