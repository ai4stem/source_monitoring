"""
analyze_2x2_semantic.py  —  re-run the 2x2 analyses with meaning-level support
=====================================================================================
Same hidden states (outputs/hidden2x2_<model>.npz), same probes and same matching code
as analyze_2x2_matched.py / analyze_2x2.py (imported, not re-implemented). Only the
SUPPORT variable changes, taken from outputs/support_semantic_<model>.npz:

    string  exact-match string support  (the original; reproduces the published JSONs)
    anchor  meaning-level, correctness-blind clustering        (primary)
    gold    meaning-level, correct answers share a cluster     (sensitivity)

For every model and support variant it reports
  matched     hidden-state correctness AUROC on 1:1 caliper-matched pairs, with the
              support-only / entropy-only AUROC and support gap on the same set
  terciles    correctness AUROC within the lower / upper support tercile, with the
              support-only AUROC inside each stratum
  unanimous   correctness AUROC among facts whose six answers all agree (support = 1),
              i.e. correct versus stably-incorrect in the strict sense, and among the
              rest (support < 1)
  descriptives  support-only AUROC for correctness on all facts (the confound),
              median-split cell counts, share of facts with support = 1

CPU only. One process per model.
OUTPUT  outputs/analysis_2x2_semantic.json     (existing JSONs are not touched)

Usage:
    python analyze_2x2_semantic.py --model all --jobs 13
"""
from __future__ import annotations
import argparse, glob, json, os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np
from analyze_2x2_matched import caliper_match, hidden_auroc_over_depth, _auroc
from analyze_2x2 import best_over_fracs, _balance

VARIANTS = {"string": ("support_string", None),
            "anchor": ("support_anchor", "entropy_anchor"),
            "gold": ("support_gold", "entropy_gold")}


def _scalar_auroc(mask_a, mask_b, scalar):
    ia = np.where(mask_a)[0]; ib = np.where(mask_b)[0]
    sel = _balance(ia, ib)
    if sel is None:
        return None
    sa = set(ia.tolist())
    y = [1 if i in sa else 0 for i in sel]
    return round(_auroc(list(scalar[sel]), y), 4)


def _r(a):
    return None if a is None else round(float(a), 4)


def one_variant(d, fracs, support, entropy, correct, caliper):
    corr = correct == 1; inc = correct == 0
    ci = np.where(corr)[0]; ii = np.where(inc)[0]
    out = {}
    # ---- descriptives
    med = np.median(support); hi = support > med
    out["support_mean"] = _r(support.mean())
    out["support_median"] = _r(med)
    out["frac_unanimous"] = _r((support >= 0.999).mean())
    out["support_only_AUROC_all"] = _r(_auroc(list(support), list(correct)))
    out["cells_median_split"] = {"correct_low": int((corr & ~hi).sum()), "correct_high": int((corr & hi).sum()),
                                 "incorrect_low": int((inc & ~hi).sum()), "incorrect_high": int((inc & hi).sum())}
    # ---- matched
    mp, mn = caliper_match(support[ci], ci, support[ii], ii, caliper)
    gi = np.concatenate([mp, mn]); y = np.array([1] * len(mp) + [0] * len(mn))
    out["matched"] = {
        "n_pairs": int(len(mp)),
        "truth_AUROC": hidden_auroc_over_depth(d, fracs, gi, y),
        "support_only_AUROC": _r(_auroc(list(support[gi]), list(y))),
        "entropy_only_AUROC": _r(_auroc(list(-entropy[gi]), list(y))),
        "support_gap": _r(support[mp].mean() - support[mn].mean()),
    }
    # ---- terciles (same rule as analyze_2x2.py)
    lo_t = np.percentile(support, 33.3); hi_t = np.percentile(support, 66.7)
    low = support <= lo_t; high = support >= hi_t
    a_lo, _, n_lo = best_over_fracs(d, fracs, low & corr, low & inc)
    a_hi, _, n_hi = best_over_fracs(d, fracs, high & corr, high & inc)
    out["terciles"] = {
        "cutpoints": [_r(lo_t), _r(hi_t)],
        "low_truth_AUROC": _r(a_lo), "low_n_per_class": n_lo,
        "high_truth_AUROC": _r(a_hi), "high_n_per_class": n_hi,
        "low_support_only_AUROC": _scalar_auroc(low & corr, low & inc, support),
        "high_support_only_AUROC": _scalar_auroc(high & corr, high & inc, support),
    }
    # ---- unanimous (all six answers agree) versus the rest
    una = support >= 0.999
    a_u, _, n_u = best_over_fracs(d, fracs, una & corr, una & inc)
    a_n, _, n_n = best_over_fracs(d, fracs, ~una & corr, ~una & inc)
    out["unanimous"] = {
        "truth_AUROC": _r(a_u), "n_per_class": n_u,
        "n_correct": int((una & corr).sum()), "n_incorrect": int((una & inc).sum()),
        "rest_truth_AUROC": _r(a_n), "rest_n_per_class": n_n,
        "rest_support_only_AUROC": _scalar_auroc(~una & corr, ~una & inc, support),
    }
    return out


def run_model(key, caliper=0.05):
    hp = f"outputs/hidden2x2_{key}.npz"; sp = f"outputs/support_semantic_{key}.npz"
    if not (os.path.exists(hp) and os.path.exists(sp)):
        return None
    z = np.load(hp); s = np.load(sp)
    fracs = [float(f) for f in z["fracs"]]
    d = {f"X_{f}": z[f"X_{f}"] for f in fracs}          # load once, reuse across variants
    correct = z["correct"].astype(int)
    assert np.array_equal(correct, s["correct"].astype(int))
    res = {"model": key, "n": int(len(correct)), "accuracy": _r(correct.mean()),
           "frac_long_answers_unclustered": _r(s["frac_singleton_long"])}
    for name, (skey, ekey) in VARIANTS.items():
        support = s[skey].astype(float)
        entropy = z["entropy"].astype(float) if ekey is None else s[ekey].astype(float)
        res[name] = one_variant(d, fracs, support, entropy, correct, caliper)
    m = {k: res[k]["matched"]["truth_AUROC"] for k in VARIANTS}
    print(f"[done] {key}: matched string {m['string']} | anchor {m['anchor']} | gold {m['gold']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all")
    ap.add_argument("--jobs", type=int, default=13)
    ap.add_argument("--caliper", type=float, default=0.05)
    ap.add_argument("--out", default="outputs/analysis_2x2_semantic.json")
    args = ap.parse_args()
    keys = sorted(os.path.basename(p)[10:-4] for p in glob.glob("outputs/hidden2x2_*.npz")) \
        if args.model == "all" else args.model.split(",")
    if args.jobs > 1 and len(keys) > 1:
        from multiprocessing import Pool
        with Pool(min(args.jobs, len(keys))) as pool:
            rows = pool.starmap(run_model, [(k, args.caliper) for k in keys])
    else:
        rows = [run_model(k, args.caliper) for k in keys]
    rows = [r for r in rows if r]
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"saved -> {args.out}")
    import statistics as st
    print(f"\n{'variant':8s} {'matched':>8s} {'sup-only':>9s} {'low':>7s} {'high':>7s} {'unanim.':>8s} {'rest':>7s}")
    for v in VARIANTS:
        def g(f):
            xs = [x for x in (f(r[v]) for r in rows) if x is not None]
            return st.mean(xs) if xs else float('nan')
        print(f"{v:8s} {g(lambda o: o['matched']['truth_AUROC']):8.3f} "
              f"{g(lambda o: o['matched']['support_only_AUROC']):9.3f} "
              f"{g(lambda o: o['terciles']['low_truth_AUROC']):7.3f} "
              f"{g(lambda o: o['terciles']['high_truth_AUROC']):7.3f} "
              f"{g(lambda o: o['unanimous']['truth_AUROC']):8.3f} "
              f"{g(lambda o: o['unanimous']['rest_truth_AUROC']):7.3f}")


if __name__ == "__main__":
    main()
