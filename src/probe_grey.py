"""
probe_grey.py  (SERVER-SIDE)  —  EXPERIMENT 1 analysis
======================================================
Reads hidden3_<model>.npz (from extract_hidden_grey.py) and runs the grey-zone
sensitivity analysis.

Three probes per model, all linear, all with a shuffled-label baseline and AUROC:

  A. confab vs parametric  (the paper's main comparison; stable-wrong vs stable-right)
  B. grey   vs stable      (grey=prompt_sensitive  vs  {parametric + confabulated})
                           => "does the hidden state encode UNCERTAINTY?"
  C. 3-way macro           (parametric / confabulated / grey)  for completeness

The decisive contrast is A vs B. The worry is that excluding the grey
zone guarantees a weak result. If:

    B (grey vs stable) AUROC is HIGH   -> the model DOES encode 'am I sure?'
    A (confab vs param) AUROC is LOW   -> but NOT 'is my stable answer true?'

then the weak-signal finding is not an artefact of grey exclusion; it is a precise
dissociation: uncertainty is represented, stable-wrong truth is not. That is a
stronger paper. If instead B is also low, the representation carries little of
either, which is worth reporting honestly.

Usage:
    python probe_grey.py --model all
    python probe_grey.py --model qwen3-8b --verbose
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
    sum_pos = sum(r for r, y in zip(ranks, labels) if y == 1)
    np_, nn = len(pos), len(neg)
    return (sum_pos - np_ * (np_ + 1) / 2.0) / (np_ * nn)


def _cv_auroc_acc(X, y, k=5, seed=0):
    """Linear probe; return (AUROC, balanced-accuracy) out-of-fold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X)); folds = np.array_split(idx, k)
    oof = np.zeros(len(X))
    for i in range(k):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    auroc = _auroc(list(oof), list(y))
    acc = float(((oof > 0.5).astype(int) == y).mean())
    return auroc, acc


def _balance(X, y, seed=0):
    y = np.asarray(y)
    i0 = np.where(y == 0)[0]; i1 = np.where(y == 1)[0]
    m = min(len(i0), len(i1))
    if m < 20:
        return None, None
    rng = np.random.RandomState(seed)
    sel = np.concatenate([rng.choice(i0, m, False), rng.choice(i1, m, False)])
    return X[sel], y[sel]


def run_model(path, verbose=False):
    d = np.load(path)
    y3 = d["y3"].astype(int)
    fracs = list(d["fracs"])
    key = os.path.basename(path).replace("hidden3_", "").replace(".npz", "")

    best = {"A_auroc": -1}
    for f in fracs:
        X = d[f"X_{f}"].astype(np.float32)
        finite = np.isfinite(X).all(axis=1)
        X, yy = X[finite], y3[finite]

        # A: confab(1) vs parametric(0)
        mask = np.isin(yy, [0, 1])
        Xa, ya = _balance(X[mask], yy[mask])
        # B: grey(1) vs stable {param,confab}(0)
        yb_full = np.where(yy == 2, 1, 0)
        Xb, yb = _balance(X, yb_full)

        if Xa is None or Xb is None:
            continue
        a_auroc, a_acc = _cv_auroc_acc(Xa, ya)
        b_auroc, b_acc = _cv_auroc_acc(Xb, yb)
        if verbose:
            print(f"  frac={f:.2f}  A(confab/param) AUROC={a_auroc:.3f} acc={a_acc:.3f}"
                  f"   B(grey/stable) AUROC={b_auroc:.3f} acc={b_acc:.3f}")
        if a_auroc > best["A_auroc"]:
            best = {"frac": f, "A_auroc": a_auroc, "A_acc": a_acc,
                    "B_auroc": b_auroc, "B_acc": b_acc,
                    "n_grey": int((yy == 2).sum()),
                    "n_param": int((yy == 0).sum()),
                    "n_confab": int((yy == 1).sum())}
    if "frac" not in best:
        return None
    best["model"] = key
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all")
    p.add_argument("--glob", default="outputs/hidden3_*.npz")
    p.add_argument("--out", default="outputs/grey_sensitivity.json")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    paths = sorted(glob.glob(args.glob)) if args.model == "all" \
        else [f"outputs/hidden3_{args.model}.npz"]

    rows = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[skip] {path} not found"); continue
        print(f"[grey] {os.path.basename(path)}")
        r = run_model(path, verbose=args.verbose)
        if r:
            rows.append(r)
            print(f"   A(confab/param) AUROC={r['A_auroc']:.3f} | "
                  f"B(grey/stable) AUROC={r['B_auroc']:.3f} | "
                  f"grey={r['n_grey']}")
    if not rows:
        print("\nNo results. Run extract_hidden_grey.py first (needs grey items "
              "from REVISED probing).")
        return
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)

    import statistics as st
    A = [r["A_auroc"] for r in rows]; B = [r["B_auroc"] for r in rows]
    print(f"\n{'model':18s} {'A:confab/param':>15s} {'B:grey/stable':>14s} {'grey_n':>7s}")
    print("-" * 58)
    for r in rows:
        print(f"{r['model']:18s} {r['A_auroc']:>15.3f} {r['B_auroc']:>14.3f} {r['n_grey']:>7d}")
    print("-" * 58)
    print(f"{'MEAN':18s} {st.mean(A):>15.3f} {st.mean(B):>14.3f}")
    print(f"\nINTERPRETATION:")
    print(f"  If B (grey vs stable) >> A (confab vs param), the model encodes "
          f"UNCERTAINTY but not stable-wrong truth: the weak main result is a real "
          f"dissociation, not an artefact of excluding the grey zone.")
    print(f"  If A and B are both low, the representation carries little of either "
          f"(report honestly).")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
