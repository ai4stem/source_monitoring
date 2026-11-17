"""
scale_correctness.py  —  Contribution 4: does scale change correctness legibility?
==================================================================================
Tests whether the STABILITY-CONTROLLED correctness separability changes with model
size. Uses the numbers already produced by the 2x2 analyses; no model re-runs.

We regress three correctness-legibility measures on log2(model size):
  (M) caliper-matched correctness AUROC        (from analysis_2x2_matched.json)
  (L) low-stability correctness AUROC          (from analysis_2x2.json, Q2 low)
  (H) high-stability correctness AUROC         (from analysis_2x2.json, Q2 high)

For each we report:
  - OLS slope per doubling, 95% CI, p, R^2
  - Spearman rho (rank, robust to n=13)
  - TOST equivalence test against a bound (default 0.02 AUROC / doubling) to state
    scale-invariance ACTIVELY, not just as a failed NHST
  - family-wise slopes (Qwen3, Llama, Phi, Mistral, Gemma) to address the
    non-independence of 5 Qwen3 / 3 Llama models

Because sizes repeat within families, OLS treating 13 as independent is only a
first pass; the family slopes and the equivalence test are the honest summary.

INPUT : outputs/analysis_2x2_matched.json, outputs/analysis_2x2.json
        (sizes taken from a small built-in map; edit SIZE if keys differ)
OUTPUT: outputs/scale_correctness.json  and a printed table
Usage:
    python scale_correctness.py
    python scale_correctness.py --bound 0.02
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from scipy import stats

SIZE = {
    "llama32-1b": 1.0, "qwen3-1.7b": 1.7, "llama32-3b": 3.0, "phi4-mini": 3.8,
    "phi3-mini-legacy": 3.8, "qwen3-4b": 4.0, "mistral-7b": 7.0, "qwen3-8b": 8.0,
    "llama3-8b-legacy": 8.0, "gemma3-12b": 12.0, "mistral-nemo-12b": 12.0,
    "qwen3-14b": 14.0, "qwen3-32b": 32.0,
}
FAMILY = {
    "llama32-1b": "llama", "llama32-3b": "llama", "llama3-8b-legacy": "llama",
    "qwen3-1.7b": "qwen3", "qwen3-4b": "qwen3", "qwen3-8b": "qwen3",
    "qwen3-14b": "qwen3", "qwen3-32b": "qwen3", "phi4-mini": "phi",
    "phi3-mini-legacy": "phi", "mistral-7b": "mistral", "mistral-nemo-12b": "mistral",
    "gemma3-12b": "gemma",
}


def regress(sizes, vals):
    x = np.log2(np.array(sizes)); y = np.array(vals)
    sl, ic, r, p, se = stats.linregress(x, y)
    df = len(x) - 2
    tcrit = stats.t.ppf(0.975, df)
    ci = (sl - tcrit * se, sl + tcrit * se)
    rho, prho = stats.spearmanr(sizes, vals)
    return {"slope": sl, "se": se, "ci": ci, "p": p, "r2": r**2,
            "spearman_rho": rho, "spearman_p": prho, "df": df}


def tost(slope, se, df, bound):
    t_lo = (slope - (-bound)) / se
    t_hi = (slope - bound) / se
    p_lo = 1 - stats.t.cdf(t_lo, df)
    p_hi = stats.t.cdf(t_hi, df)
    return max(p_lo, p_hi)


def family_slopes(keys, sizes, vals):
    out = {}
    by = {}
    for k, s, v in zip(keys, sizes, vals):
        by.setdefault(FAMILY[k], []).append((s, v))
    for fam, pts in by.items():
        if len(pts) < 3:
            out[fam] = None; continue
        xs = np.log2([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        sl = stats.linregress(xs, ys).slope
        out[fam] = round(float(sl), 4)
    return out


def load_measure(matched_path, tercile_path):
    matched = {r["model"]: r for r in json.load(open(matched_path))}
    tercile = {r["model"]: r for r in json.load(open(tercile_path))}
    keys = [k for k in SIZE if k in matched]
    M = [(k, SIZE[k], matched[k]["Q2_truth_matched_AUROC"]) for k in keys
         if matched[k].get("Q2_truth_matched_AUROC") is not None]
    L = [(k, SIZE[k], tercile[k]["Q2_truth_within_lowsupport_AUROC"]) for k in keys
         if k in tercile and tercile[k].get("Q2_truth_within_lowsupport_AUROC") is not None]
    H = [(k, SIZE[k], tercile[k]["Q2_truth_within_highsupport_AUROC"]) for k in keys
         if k in tercile and tercile[k].get("Q2_truth_within_highsupport_AUROC") is not None]
    return M, L, H


def analyze(name, triples, bound):
    keys = [t[0] for t in triples]; sizes = [t[1] for t in triples]; vals = [t[2] for t in triples]
    reg = regress(sizes, vals)
    p_tost = tost(reg["slope"], reg["se"], reg["df"], bound)
    fam = family_slopes(keys, sizes, vals)
    import statistics as st
    print(f"\n=== {name} (n={len(triples)}) ===")
    print(f"  mean AUROC = {st.mean(vals):.3f}")
    print(f"  slope = {reg['slope']:+.4f}/doubling  95% CI [{reg['ci'][0]:+.4f}, {reg['ci'][1]:+.4f}]  p={reg['p']:.3f}  R2={reg['r2']:.3f}")
    print(f"  Spearman rho = {reg['spearman_rho']:+.3f} (p={reg['spearman_p']:.3f})")
    print(f"  TOST (bound ±{bound}/doubling): p={p_tost:.4f}  "
          f"{'EQUIVALENT to zero (scale-invariant)' if p_tost<0.05 else 'not equivalent'}")
    print(f"  family slopes: {fam}")
    return {"measure": name, "n": len(triples), "mean": round(st.mean(vals), 4),
            "slope": round(reg["slope"], 5), "ci": [round(c, 5) for c in reg["ci"]],
            "p": round(reg["p"], 4), "r2": round(reg["r2"], 4),
            "spearman_rho": round(reg["spearman_rho"], 4),
            "spearman_p": round(reg["spearman_p"], 4),
            "tost_bound": bound, "tost_p": round(p_tost, 4),
            "family_slopes": fam,
            "per_model": [{"model": k, "size": s, "auroc": v} for k, s, v in triples]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched", default="outputs/analysis_2x2_matched.json")
    ap.add_argument("--tercile", default="outputs/analysis_2x2.json")
    ap.add_argument("--bound", type=float, default=0.02)
    ap.add_argument("--out", default="outputs/scale_correctness.json")
    args = ap.parse_args()
    for p in (args.matched, args.tercile):
        if not os.path.exists(p):
            print(f"[error] missing {p}"); return

    M, L, H = load_measure(args.matched, args.tercile)
    results = []
    results.append(analyze("caliper-matched correctness", M, args.bound))
    results.append(analyze("low-stability correctness", L, args.bound))
    results.append(analyze("high-stability correctness", H, args.bound))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nINTERPRETATION:")
    print(f"  If slopes are near zero and TOST is significant, scale does not change")
    print(f"  correctness legibility (Contribution 4). The gap between low- and high-")
    print(f"  stability correctness (Contribution 3) persists across size if BOTH L and")
    print(f"  H have flat slopes and H stays below L at every size.")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
