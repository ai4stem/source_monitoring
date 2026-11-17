"""
analyze_2x2_levels.py  —  leakage-free regime contrasts for the meaning-level support
=====================================================================================
The unanimous stratum (support = 1) holds support constant, but "all other facts" does
not: inside it support still predicts correctness, so its correctness AUROC is not a
like-for-like comparison. This script adds the comparisons that are:

  overall          correctness AUROC on all facts, balanced, no control (reference)
  matched_rest     caliper-matched pairs drawn only from non-unanimous facts (support < 1)
  by_level         correctness AUROC within each exact support level (support takes a
                   handful of discrete values), so support is constant within a stratum

Same hidden states and probe code as the other 2x2 scripts (imported). CPU only.
OUTPUT  outputs/analysis_2x2_levels.json
Usage:  python analyze_2x2_levels.py --model all --jobs 13 [--variant anchor]
"""
from __future__ import annotations
import argparse, glob, json, os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
from analyze_2x2_matched import caliper_match, hidden_auroc_over_depth, _auroc
from analyze_2x2 import best_over_fracs

MIN_PER_CLASS = 50


def run_model(key, variant="anchor", caliper=0.05):
    z = np.load(f"outputs/hidden2x2_{key}.npz"); s = np.load(f"outputs/support_semantic_{key}.npz")
    fracs = [float(f) for f in z["fracs"]]
    d = {f"X_{f}": z[f"X_{f}"] for f in fracs}
    correct = z["correct"].astype(int); support = s[f"support_{variant}"].astype(float)
    corr = correct == 1; inc = ~corr
    res = {"model": key, "variant": variant}
    a, _, n = best_over_fracs(d, fracs, corr, inc)
    res["overall"] = {"truth_AUROC": round(float(a), 4), "n_per_class": int(n)}
    rest = support < 0.999
    ci = np.where(corr & rest)[0]; ii = np.where(inc & rest)[0]
    mp, mn = caliper_match(support[ci], ci, support[ii], ii, caliper)
    gi = np.concatenate([mp, mn]); y = np.array([1] * len(mp) + [0] * len(mn))
    res["matched_rest"] = {"n_pairs": int(len(mp)), "truth_AUROC": hidden_auroc_over_depth(d, fracs, gi, y),
                           "support_only_AUROC": round(float(_auroc(list(support[gi]), list(y))), 4)}
    levels = []
    lv = np.round(support, 3)
    for v in sorted(set(lv.tolist())):
        m = lv == v
        nc, ni = int((m & corr).sum()), int((m & inc).sum())
        row = {"support": v, "n_correct": nc, "n_incorrect": ni, "truth_AUROC": None}
        if min(nc, ni) >= MIN_PER_CLASS:
            a, _, n = best_over_fracs(d, fracs, m & corr, m & inc)
            row["truth_AUROC"] = round(float(a), 4); row["n_per_class"] = int(n)
        levels.append(row)
    res["by_level"] = levels
    print(f"[done] {key}: overall {res['overall']['truth_AUROC']} | matched_rest {res['matched_rest']['truth_AUROC']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all"); ap.add_argument("--jobs", type=int, default=13)
    ap.add_argument("--variant", default="anchor")
    ap.add_argument("--out", default="outputs/analysis_2x2_levels.json")
    args = ap.parse_args()
    keys = sorted(os.path.basename(p)[10:-4] for p in glob.glob("outputs/hidden2x2_*.npz")) \
        if args.model == "all" else args.model.split(",")
    from multiprocessing import Pool
    with Pool(min(args.jobs, len(keys))) as pool:
        rows = pool.starmap(run_model, [(k, args.variant) for k in keys])
    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
