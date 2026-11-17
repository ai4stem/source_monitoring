"""
analyze_2x2.py  (SERVER-SIDE)  —  support vs truth analysis
=======================================================================
Uses hidden2x2_<model>.npz (support, correct, entropy + hidden states) to answer
two orthogonal questions, with controls so the result cannot be dismissed as circular.

Q1 (truth fixed, vary support):  within CORRECT answers, can the hidden state
    tell high-support from low-support?   -> AUROC_support|correct
    (also within incorrect, as a robustness check)

Q2 (support fixed, vary truth):  MATCHING on support, can the hidden state tell
    correct from incorrect?               -> AUROC_truth|support-matched
    We match by taking the middle band of support shared by both classes and, in
    addition, report truth separability inside the LOW-support and HIGH-support
    strata separately.

The decisive comparison: if support stays clearly decodable (Q1 high) while truth
becomes much weaker after matching (Q2 low), the representation is organised more
by support than by truth.

CONTROLS built in:
  * support split uses TERCILES (low third vs high third) for a clean contrast.
  * SUPPORT-ONLY BASELINE: a classifier using only the scalar support value (and
    only entropy) is reported next to the hidden-state probe, so we can show the
    hidden state does NOT merely re-read support/entropy. For Q2 we also give the
    truth AUROC of a support-only and entropy-only predictor (should be ~0.5 after
    matching) to prove the truth test is not contaminated.
  * ENTROPY/FAMILIARITY note: entropy is stored; Q2 matches on support, and we
    report the residual support/entropy gap between the matched classes.

OUTPUT: outputs/analysis_2x2.json
Usage:
    python analyze_2x2.py --model all --verbose
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
    if len(np.unique(y)) < 2:
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


def _balance(idx0, idx1, seed=0):
    m = min(len(idx0), len(idx1))
    if m < 25:
        return None
    rng = np.random.RandomState(seed)
    return np.concatenate([rng.choice(idx0, m, False), rng.choice(idx1, m, False)])


def best_over_fracs(d, fracs, mask_pos, mask_neg, seed=0):
    """AUROC separating mask_pos(1) vs mask_neg(0), balanced, best over depth."""
    ip = np.where(mask_pos)[0]; ineg = np.where(mask_neg)[0]
    sel = _balance(ip, ineg, seed)
    if sel is None:
        return None, None, None
    y = np.array([1 if i in set(ip) else 0 for i in sel])
    best_a, best_f = -1, None
    for f in fracs:
        X = d[f"X_{f}"].astype(np.float32)[sel]
        fin = np.isfinite(X).all(axis=1)
        a = _cv_auroc(X[fin], y[fin], seed=seed)
        if a == a and a > best_a:
            best_a, best_f = a, f
    return best_a, best_f, len(sel) // 2


def run_model(path, verbose=False):
    d = np.load(path)
    fracs = [float(f) for f in d["fracs"]]
    support = d["support"].astype(float)
    correct = d["correct"].astype(int)
    entropy = d["entropy"].astype(float)
    key = os.path.basename(path).replace("hidden2x2_", "").replace(".npz", "")

    # terciles of support
    lo_t = np.percentile(support, 33.3)
    hi_t = np.percentile(support, 66.7)
    low = support <= lo_t
    high = support >= hi_t
    corr = correct == 1
    inc = correct == 0

    res = {"model": key, "n": int(len(support)),
           "support_terciles": [round(lo_t, 3), round(hi_t, 3)]}

    # ---- Q1: truth fixed (correct only), vary support (high vs low) ----
    a_sup_corr, f1, n1 = best_over_fracs(d, fracs, corr & high, corr & low)
    a_sup_inc, _, _ = best_over_fracs(d, fracs, inc & high, inc & low)
    # support-only + entropy-only baselines for Q1 (should be ~1.0 since split IS support)
    res["Q1_support_within_correct_AUROC"] = None if a_sup_corr is None else round(a_sup_corr, 4)
    res["Q1_support_within_incorrect_AUROC"] = None if a_sup_inc is None else round(a_sup_inc, 4)
    res["Q1_n_per_class"] = n1

    # ---- Q2: support fixed, vary truth ----
    # (2a) within LOW-support stratum: correct vs incorrect
    a_truth_low, f2l, n2l = best_over_fracs(d, fracs, low & corr, low & inc)
    # (2b) within HIGH-support stratum: correct vs incorrect
    a_truth_high, f2h, n2h = best_over_fracs(d, fracs, high & corr, high & inc)
    res["Q2_truth_within_lowsupport_AUROC"] = None if a_truth_low is None else round(a_truth_low, 4)
    res["Q2_truth_within_highsupport_AUROC"] = None if a_truth_high is None else round(a_truth_high, 4)
    res["Q2_n_low"] = n2l; res["Q2_n_high"] = n2h

    # ---- baselines proving Q2 is not contaminated by support/entropy ----
    # after stratifying by support tercile, does support alone predict truth? (want ~0.5)
    def scalar_auroc(mask_a, mask_b, scalar):
        ia = np.where(mask_a)[0]; ib = np.where(mask_b)[0]
        sel = _balance(ia, ib)
        if sel is None:
            return None
        y = np.array([1 if i in set(ia) else 0 for i in sel])
        return round(_auroc(list(scalar[sel]), list(y)), 4)
    res["Q2_support_only_lowstratum"] = scalar_auroc(low & corr, low & inc, support)
    res["Q2_support_only_highstratum"] = scalar_auroc(high & corr, high & inc, support)
    res["Q2_entropy_only_lowstratum"] = scalar_auroc(low & corr, low & inc, -entropy)

    # residual support/entropy gap between matched truth classes (diagnostics)
    def gap(mask_a, mask_b, arr):
        if mask_a.sum() < 5 or mask_b.sum() < 5:
            return None
        return round(float(arr[mask_a].mean() - arr[mask_b].mean()), 4)
    res["support_gap_low_stratum"] = gap(low & corr, low & inc, support)
    res["support_gap_high_stratum"] = gap(high & corr, high & inc, support)

    if verbose:
        print(f"  {key}:")
        print(f"    Q1 support|correct   AUROC={res['Q1_support_within_correct_AUROC']}  "
              f"(support IS decodable within truth)")
        print(f"    Q2 truth|low-support  AUROC={res['Q2_truth_within_lowsupport_AUROC']}  "
              f"(truth after matching, low stratum)")
        print(f"    Q2 truth|high-support AUROC={res['Q2_truth_within_highsupport_AUROC']}  "
              f"(truth after matching, high stratum)")
        print(f"    [baseline] support-only predicts truth (low): "
              f"{res['Q2_support_only_lowstratum']} (want ~0.5)")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all")
    p.add_argument("--glob", default="outputs/hidden2x2_*.npz")
    p.add_argument("--out", default="outputs/analysis_2x2.json")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    paths = sorted(glob.glob(args.glob)) if args.model == "all" \
        else [f"outputs/hidden2x2_{args.model}.npz"]

    rows = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[skip] {path} not found"); continue
        print(f"[2x2] {os.path.basename(path)}")
        r = run_model(path, verbose=args.verbose)
        if r:
            rows.append(r)
    if not rows:
        print("\nNo results. Run extract_hidden_2x2.py first."); return
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)

    import statistics as st
    def m(k):
        vals = [r[k] for r in rows if r.get(k) is not None]
        return st.mean(vals) if vals else float("nan")

    print(f"\n{'model':18s} {'Q1 sup|corr':>11s} {'Q2 truth|lo':>11s} {'Q2 truth|hi':>11s} "
          f"{'sup-only|lo':>11s}")
    print("-" * 66)
    for r in rows:
        print(f"{r['model']:18s} "
              f"{str(r['Q1_support_within_correct_AUROC']):>11s} "
              f"{str(r['Q2_truth_within_lowsupport_AUROC']):>11s} "
              f"{str(r['Q2_truth_within_highsupport_AUROC']):>11s} "
              f"{str(r['Q2_support_only_lowstratum']):>11s}")
    print("-" * 66)
    print(f"{'MEAN':18s} {m('Q1_support_within_correct_AUROC'):>11.3f} "
          f"{m('Q2_truth_within_lowsupport_AUROC'):>11.3f} "
          f"{m('Q2_truth_within_highsupport_AUROC'):>11.3f} "
          f"{m('Q2_support_only_lowstratum'):>11.3f}")
    print(f"\nINTERPRETATION:")
    print(f"  Q1 (support within correct) high  AND  Q2 (truth within a support")
    print(f"  stratum) low  =>  the hidden state is organised more by support than")
    print(f"  by truth, WITHOUT the circularity, because support here is estimated")
    print(f"  leave-one-paraphrase-out and Q2 holds support fixed. The support-only")
    print(f"  baseline for Q2 should sit near 0.5, proving the truth test is not")
    print(f"  just re-reading support.")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
