"""
analyze_2x2_primary.py  —  PRIMARY 2x2 analysis: matching on support AND dataset
=====================================================================================
Support-only matching leaves the source dataset unbalanced between correct and incorrect
facts (TriviaQA is much easier than PopQA), and the dataset indicator alone then predicts
correctness on the matched pairs (AUROC ~0.62). All stability-controlled comparisons are
therefore formed WITHIN dataset here: a correct and an incorrect fact are paired only if
they come from the same dataset and their support lies within the caliper.

Per model and support variant (anchor = primary; gold, string = sensitivity):
  overall          all facts, balanced, no control                       (reference)
  matched          pairs within dataset and caliper, pooled
  matched_trivia   the same, TriviaQA only (no relation-type templates)
  matched_popqa    the same, PopQA only
  matched_rest     pairs drawn from non-unanimous facts only
  unanimous        pairs drawn from unanimous facts only (support = 1), within dataset
For the anchor variant also:
  seeds            matched AUROC over five seeds (matching order and folds)
  nested           matched AUROC with depth chosen inside each training fold
Leak checks on every matched set: support-only, dataset-only and entropy-only AUROC.

Same hidden states, probe and matching code as the other scripts. CPU only.
OUTPUT  outputs/analysis_2x2_primary.json
Usage:  python analyze_2x2_primary.py --jobs 13
"""
from __future__ import annotations
import argparse, glob, json, os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
from analyze_2x2_matched import caliper_match, _auroc, _cv_auroc
from analyze_2x2 import best_over_fracs
from analyze_2x2_dataset import dataset_ids

VARIANTS = {"anchor": ("support_anchor", "entropy_anchor"), "gold": ("support_gold", "entropy_gold"),
            "string": ("support_string", None)}


def _r(a):
    return None if a is None else round(float(a), 4)


def run_model(key, caliper=0.05):
    z = np.load(f"outputs/hidden2x2_{key}.npz"); s = dict(np.load(f"outputs/support_semantic_{key}.npz"))
    if os.path.exists(f"outputs/support_nli_{key}.npz"):          # entailment-based clusterings (nli_cluster_support.py)
        s.update({k: v for k, v in np.load(f"outputs/support_nli_{key}.npz").items() if k != "correct"})
    fracs = [float(f) for f in z["fracs"]]
    X = {f: z[f"X_{f}"].astype(np.float32) for f in fracs}
    correct = z["correct"].astype(int); ds = dataset_ids(key, len(correct))
    res = {"model": key, "n": int(len(correct))}
    a, _, n = best_over_fracs({f"X_{f}": z[f"X_{f}"] for f in fracs}, fracs, correct == 1, correct == 0)
    res["overall"] = {"truth_AUROC": _r(a), "n_per_class": int(n)}

    def pairs(support, mask, seed=0):
        mp, mn = [], []
        for v in (0, 1):                                   # match within dataset
            ci = np.where((correct == 1) & mask & (ds == v))[0]
            ii = np.where((correct == 0) & mask & (ds == v))[0]
            if len(ci) and len(ii):
                p, q = caliper_match(support[ci], ci, support[ii], ii, caliper, seed=seed)
                mp.append(p); mn.append(q)
        mp = np.concatenate(mp) if mp else np.array([], int); mn = np.concatenate(mn) if mn else np.array([], int)
        return mp, mn

    def score(support, entropy, mask, seed=0, min_pairs=40):
        mp, mn = pairs(support, mask, seed)
        out = {"n_pairs": int(len(mp)), "truth_AUROC": None}
        if len(mp) < min_pairs:
            return out
        gi = np.concatenate([mp, mn]); y = np.array([1] * len(mp) + [0] * len(mn))
        per = {f: _cv_auroc(X[f][gi], y, seed=seed) for f in fracs}
        out.update(truth_AUROC=_r(max(per.values())), best_depth=max(per, key=per.get),
                   support_only_AUROC=_r(_auroc(list(support[gi]), list(y))),
                   dataset_only_AUROC=_r(_auroc(list(ds[gi].astype(float)), list(y))),
                   entropy_only_AUROC=_r(_auroc(list(-entropy[gi]), list(y))))
        return out

    allm = np.ones(len(correct), bool)
    for name, (sk, ek) in VARIANTS.items():
        support = s[sk].astype(float)
        entropy = z["entropy"].astype(float) if ek is None else s[ek].astype(float)
        una = support >= 0.999
        o = {"matched": score(support, entropy, allm),
             "matched_trivia": score(support, entropy, ds == 0),
             "matched_popqa": score(support, entropy, ds == 1),
             "matched_rest": score(support, entropy, ~una),
             "unanimous": score(support, entropy, una)}
        if name == "anchor":
            vals = [score(support, entropy, allm, seed=sd)["truth_AUROC"] for sd in range(5)]
            o["seeds"] = {"values": vals, "mean": _r(np.mean(vals)), "sd": _r(np.std(vals, ddof=1))}
            # nested depth selection on the seed-0 matched set
            from analyze_2x2_robustness import _fit_predict, _folds
            mp, mn = pairs(support, allm, 0)
            gi = np.concatenate([mp, mn]); y = np.array([1] * len(mp) + [0] * len(mn))
            folds = _folds(len(gi), 5, 0); oof = np.zeros(len(gi))
            for i in range(5):
                te = folds[i]; tr = np.concatenate([folds[j] for j in range(5) if j != i])
                inner = {f: _cv_auroc(X[f][gi][tr], y[tr], k=4, seed=1) for f in fracs}
                fs = max(inner, key=inner.get)
                oof[te] = _fit_predict(X[fs][gi][tr], y[tr], X[fs][gi][te])
            o["nested"] = _r(_auroc(list(oof), list(y)))
        res[name] = o
    a = res["anchor"]
    print(f"[done] {key}: overall {res['overall']['truth_AUROC']} | matched {a['matched']['truth_AUROC']} "
          f"(nested {a['nested']}, seeds {a['seeds']['mean']}±{a['seeds']['sd']}) | rest {a['matched_rest']['truth_AUROC']} "
          f"| unanimous {a['unanimous']['truth_AUROC']} (n={a['unanimous']['n_pairs']}) | gold {res['gold']['matched']['truth_AUROC']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=13)
    ap.add_argument("--out", default="outputs/analysis_2x2_primary.json")
    ap.add_argument("--nli", action="store_true", help="add the four entailment-based support variants")
    args = ap.parse_args()
    if args.nli:
        for m in ("raw", "canon"):
            for c in ("strict", "loose"):
                VARIANTS[f"nli_{m}_{c}"] = (f"support_{m}_{c}", f"entropy_{m}_{c}")
    keys = sorted(os.path.basename(p)[10:-4] for p in glob.glob("outputs/hidden2x2_*.npz"))
    from multiprocessing import Pool
    with Pool(min(args.jobs, len(keys))) as pool:
        rows = pool.map(run_model, keys)
    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
