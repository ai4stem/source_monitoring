"""
split_uh_ah.py  (SERVER-SIDE)  —  Cheang UH/AH differentiation
==============================================================
Motivation: Cheang et al. (2026) split hallucinations into
  - AH (Associated Hallucination): driven by parametric associations; hidden-state
    geometry OVERLAPS factual outputs -> hard to detect.
  - UH (Unassociated Hallucination): no parametric grounding; clustered, high-
    entropy representations -> easier to detect.
Our 'confabulated' class likely mixes both. If we split it and show the hidden
state separates UH (from parametric) better than AH, we (a) differentiate from
Cheang by an independent operationalisation and (b) explain WHY our overall
truth signal is weak: the AH portion is intrinsically hard, exactly as Cheang's
mechanistic account predicts.

Operationalisation with data we already have (NO re-run):
  Cheang's AH relies on strong input-triggered parametric association -> the model
  gives a CONSISTENT wrong answer (low answer entropy across paraphrases/samples).
  UH lacks grounding -> the wrong answers are SCATTERED (high answer entropy).
  We therefore split 'confabulated' by the answer_entropy already stored in each
  fact's stability record:
      AH  = confabulated with answer_entropy <= low_pct percentile
      UH  = confabulated with answer_entropy >= high_pct percentile
  (a gap in the middle keeps the two groups clean).

We then reuse the extracted hidden states and, per model, measure how well a
linear probe separates AH-vs-parametric and UH-vs-parametric (AUROC). Prediction
(consistent with Cheang): UH is separable, AH is much harder.

INPUT : outputs/probed_<model>.jsonl  (revised probing, has answer_entropy)
        outputs/hidden_<model>.npz     (2-way hidden states, has y and X_<frac>)
        -- but hidden_ npz stores only param/confab rows in the SAME order as the
           probed facts that were resolved param/confab. To align, we re-key by
           matching. Simpler: we recompute labels from probed and re-extract is
           NOT needed because extract_hidden stored rows for confabulated too.
           We therefore need the per-row fact id. If hidden_ npz lacks ids, use
           extract_hidden_uhah.py (below) to re-extract with ids.

Because the original hidden_<model>.npz may not carry fact ids, this script reads
hidden3_<model>.npz IF it was produced by an id-carrying extractor; otherwise use
the companion extractor. To stay robust, we recompute from probed + a fresh
extract that includes ids (see extract_hidden_uhah.py).

OUTPUT: outputs/uh_ah_split.json
Usage:
    python split_uh_ah.py --model all --low_pct 33 --high_pct 66 --verbose
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
    rng = np.random.RandomState(seed); idx = rng.permutation(len(X))
    folds = np.array_split(idx, k); oof = np.zeros(len(X))
    for i in range(k):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return _auroc(list(oof), list(y))


def _balance(Xa, Xb, seed=0):
    """Balance two groups to equal size; return stacked X, y (1=first group)."""
    m = min(len(Xa), len(Xb))
    if m < 25:
        return None, None
    rng = np.random.RandomState(seed)
    ia = rng.choice(len(Xa), m, False); ib = rng.choice(len(Xb), m, False)
    X = np.vstack([Xa[ia], Xb[ib]])
    y = np.array([1] * m + [0] * m)
    return X, y


def run_model(model_key, low_pct, high_pct, verbose=False):
    npz = f"outputs/hidden_uhah_{model_key}.npz"
    if not os.path.exists(npz):
        print(f"  [skip] {model_key}: no {npz} (run extract_hidden_uhah.py first)")
        return None
    d = np.load(npz)
    fracs = [float(f) for f in d["fracs"]]
    # rows carry: label (0=param,1=confab), ent (answer_entropy)
    label = d["label"].astype(int)
    ent = d["ent"].astype(float)

    confab_ent = ent[label == 1]
    if len(confab_ent) < 60:
        print(f"  [skip] {model_key}: too few confabulated ({len(confab_ent)})")
        return None
    lo = np.percentile(confab_ent, low_pct)
    hi = np.percentile(confab_ent, high_pct)

    is_param = label == 0
    is_ah = (label == 1) & (ent <= lo)   # low entropy = consistent wrong = AH
    is_uh = (label == 1) & (ent >= hi)   # high entropy = scattered wrong = UH

    best = {"uh_auroc": -1}
    for f in fracs:
        X = d[f"X_{f}"].astype(np.float32)
        fin = np.isfinite(X).all(axis=1)
        Xp = X[is_param & fin]; Xah = X[is_ah & fin]; Xuh = X[is_uh & fin]
        Xa, ya = _balance(Xah, Xp)   # AH vs param
        Xu, yu = _balance(Xuh, Xp)   # UH vs param
        if Xa is None or Xu is None:
            continue
        a_auroc = _cv_auroc(Xa, ya)
        u_auroc = _cv_auroc(Xu, yu)
        if u_auroc > best["uh_auroc"]:
            best = {"frac": f, "ah_auroc": round(a_auroc, 4),
                    "uh_auroc": round(u_auroc, 4),
                    "n_ah": int(is_ah.sum()), "n_uh": int(is_uh.sum()),
                    "n_param": int(is_param.sum())}
    if "frac" not in best:
        print(f"  [skip] {model_key}: groups too small after balancing")
        return None
    best["model"] = model_key
    if verbose:
        print(f"  {model_key}: UH-vs-param AUROC={best['uh_auroc']:.3f} "
              f"AH-vs-param AUROC={best['ah_auroc']:.3f} "
              f"(n_ah={best['n_ah']}, n_uh={best['n_uh']})")
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all")
    p.add_argument("--low_pct", type=float, default=33)
    p.add_argument("--high_pct", type=float, default=66)
    p.add_argument("--out", default="outputs/uh_ah_split.json")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    keys = [os.path.basename(x).replace("hidden_uhah_", "").replace(".npz", "")
            for x in sorted(glob.glob("outputs/hidden_uhah_*.npz"))] \
        if args.model == "all" else [args.model]

    rows = []
    for k in keys:
        r = run_model(k, args.low_pct, args.high_pct, verbose=args.verbose)
        if r:
            rows.append(r)
    if not rows:
        print("\nNo results. Run extract_hidden_uhah.py first to produce "
              "hidden_uhah_<model>.npz (hidden states + answer_entropy + label).")
        return
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)

    import statistics as st
    uh = [r["uh_auroc"] for r in rows]; ah = [r["ah_auroc"] for r in rows]
    print(f"\n{'model':18s} {'UH-vs-param':>12s} {'AH-vs-param':>12s}")
    print("-" * 46)
    for r in rows:
        print(f"{r['model']:18s} {r['uh_auroc']:>12.3f} {r['ah_auroc']:>12.3f}")
    print("-" * 46)
    print(f"{'MEAN':18s} {st.mean(uh):>12.3f} {st.mean(ah):>12.3f}")
    print(f"\nPrediction (Cheang): UH separable, AH hard (overlaps parametric).")
    print(f"If mean UH AUROC >> mean AH AUROC, our data reproduce the UH/AH "
          f"dissociation by an independent route (answer-entropy split), and the "
          f"weak overall truth signal is explained: the AH portion is intrinsically "
          f"hard to detect, exactly as the recall-not-truth account predicts.")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
