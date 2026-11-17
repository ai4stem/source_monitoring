"""
scale_correctness_semantic.py  —  scale analysis on the meaning-level 2x2 results
=====================================================================================
Same regression, equivalence test and family slopes as scale_correctness.py (imported),
applied to outputs/analysis_2x2_primary.json (pairs matched on support AND dataset) for one
support variant (default: anchor).

Measures: uncontrolled (all facts), caliper-matched, caliper-matched among non-unanimous
facts, unanimous, plus the unmatched non-unanimous and tercile strata (which retain
support information and are kept for reference only); each regressed on log2(parameters).
The uncontrolled and matched-non-unanimous values come from analysis_2x2_levels.json.

OUTPUT  outputs/scale_correctness_semantic.json   (same schema as scale_correctness.json)
Usage:  python scale_correctness_semantic.py [--variant anchor] [--bound 0.02]
"""
from __future__ import annotations
import argparse, json
from scale_correctness import SIZE, analyze


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="outputs/analysis_2x2_primary.json")
    ap.add_argument("--variant", default="anchor")
    ap.add_argument("--bound", type=float, default=0.02)
    ap.add_argument("--out", default="outputs/scale_correctness_semantic.json")
    args = ap.parse_args()
    rows = {r["model"]: r for r in json.load(open(args.inp))}
    keys = [k for k in SIZE if k in rows]
    v = args.variant
    get = {
        "uncontrolled correctness": lambda r: r["overall"]["truth_AUROC"],
        "caliper-matched correctness": lambda r: r[v]["matched"]["truth_AUROC"],
        "matched correctness, TriviaQA": lambda r: r[v]["matched_trivia"]["truth_AUROC"],
        "matched correctness, PopQA": lambda r: r[v]["matched_popqa"]["truth_AUROC"],
        "matched non-unanimous correctness": lambda r: r[v]["matched_rest"]["truth_AUROC"],
        "unanimous correctness": lambda r: r[v]["unanimous"]["truth_AUROC"],
    }
    out = []
    for name, f in get.items():
        triples = [(k, SIZE[k], f(rows[k])) for k in keys if f(rows[k]) is not None]
        res = analyze(name, triples, args.bound)
        res["support_variant"] = v
        out.append(res)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
