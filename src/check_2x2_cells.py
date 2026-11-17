"""
check_2x2_cells.py  (SERVER-SIDE)  —  support x truth design
========================================================================
Feasibility check for the 2x2 design:

                 Low support        High support
    Correct      weakly-sup correct  strongly-sup correct
    Incorrect    weakly-sup false    strongly-sup false

The key cells that prior work (Cheang) omits are the CORRECT x LOW-SUPPORT and
INCORRECT x HIGH-SUPPORT cells. If those are too small, the design cannot run,
so we count all four cells per model before building anything.

SUPPORT is estimated independently of the target response, leave-one-paraphrase-
out (LOPO): for a fact probed with 6 paraphrases x 5 samples = 30 responses,
support is the agreement of the 25 responses from the OTHER five paraphrases,
i.e. how consistently the model produces the same answer across DIFFERENT
wordings, regardless of whether that answer is correct. This makes support a
property of the fact that is independent of any single response's hidden state.

TRUTH is whether the (modal) answer is correct, from the stored correctness.

We split support at the median (per model) into low/high, and truth by
correctness, and report the four cell counts. We use the by_paraphrase raw
samples saved by the revised probing.

INPUT : outputs/probed_<model>.jsonl  (must contain stability.by_paraphrase)
OUTPUT: prints a per-model 2x2 table; writes outputs/cells_2x2.json

Usage:
    python check_2x2_cells.py --model all
    python check_2x2_cells.py --model qwen3-8b --verbose
"""
from __future__ import annotations
import argparse, json, os, glob, re
from collections import Counter
import numpy as np


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def lopo_support(by_paraphrase):
    """Leave-one-paraphrase-out support for each paraphrase's responses.

    Returns a list of (support_value, correct_bool) per response, where the
    support for a response is computed from the OTHER paraphrases' answers only.
    support = agreement = fraction of other-paraphrase answers equal to the modal
    other-paraphrase answer (consistency across wording, correctness-independent).
    """
    # gather per-paraphrase normalized answers + correctness
    paras = []
    for pa in by_paraphrase:
        ans = [_norm(a) for a in pa.get("answers", [])]
        cor = pa.get("correct", [])
        paras.append((ans, cor))
    out = []
    n_para = len(paras)
    if n_para < 2:
        return out
    for i in range(n_para):
        # other paraphrases' answers (exclude paraphrase i entirely)
        others = []
        for j in range(n_para):
            if j == i:
                continue
            others.extend(paras[j][0])
        if not others:
            continue
        modal_count = Counter(others).most_common(1)[0][1]
        support = modal_count / len(others)   # agreement across other wordings
        # target correctness for this paraphrase: majority of its own samples
        cor_i = paras[i][1]
        if not cor_i:
            continue
        correct = sum(cor_i) / len(cor_i) >= 0.5
        out.append((support, correct))
    return out


def run_model(path, verbose=False):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    facts = [r for r in rows if r.get("item_type") == "fact"]
    # collect fact-level support (mean LOPO support) and truth (modal correctness)
    fact_support, fact_correct = [], []
    n_noraw = 0
    for r in facts:
        stab = r.get("stability") or {}
        bp = stab.get("by_paraphrase")
        if not bp:
            n_noraw += 1
            continue
        vals = lopo_support(bp)
        if not vals:
            continue
        sup = np.mean([v[0] for v in vals])
        # fact truth = majority correctness across all samples
        allcor = []
        for pa in bp:
            allcor.extend(pa.get("correct", []))
        if not allcor:
            continue
        correct = sum(allcor) / len(allcor) >= 0.5
        fact_support.append(sup)
        fact_correct.append(correct)
    if len(fact_support) < 50:
        print(f"  [skip] {os.path.basename(path)}: too few facts with raw samples "
              f"({len(fact_support)}, no_raw={n_noraw})")
        return None

    sup = np.array(fact_support); cor = np.array(fact_correct)
    med = np.median(sup)
    high = sup > med
    key = os.path.basename(path).replace("probed_", "").replace(".jsonl", "")

    cells = {
        "correct_low":   int(np.sum(cor & ~high)),
        "correct_high":  int(np.sum(cor & high)),
        "incorrect_low": int(np.sum(~cor & ~high)),
        "incorrect_high":int(np.sum(~cor & high)),
    }
    res = {"model": key, "n_facts": len(sup), "support_median": round(float(med), 3),
           "cells": cells,
           "min_cell": min(cells.values())}
    if verbose:
        print(f"  {key}: median_support={med:.3f}")
        print(f"           correct:   low={cells['correct_low']:4d}  high={cells['correct_high']:4d}")
        print(f"           incorrect: low={cells['incorrect_low']:4d}  high={cells['incorrect_high']:4d}")
        print(f"           min cell = {res['min_cell']}")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all")
    p.add_argument("--glob", default="outputs/probed_*.jsonl")
    p.add_argument("--out", default="outputs/cells_2x2.json")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    paths = sorted(glob.glob(args.glob)) if args.model == "all" \
        else [f"outputs/probed_{args.model}.jsonl"]

    rows = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[skip] {path} not found"); continue
        r = run_model(path, verbose=args.verbose)
        if r:
            rows.append(r)
    if not rows:
        print("\nNo results. Need probed_*.jsonl with by_paraphrase raw samples.")
        return
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\n{'model':18s} {'C-low':>6s} {'C-high':>7s} {'I-low':>6s} {'I-high':>7s} {'min':>5s}")
    print("-" * 56)
    for r in rows:
        c = r["cells"]
        print(f"{r['model']:18s} {c['correct_low']:>6d} {c['correct_high']:>7d} "
              f"{c['incorrect_low']:>6d} {c['incorrect_high']:>7d} {r['min_cell']:>5d}")
    mins = [r["min_cell"] for r in rows]
    print("-" * 56)
    print(f"smallest cell across all models: {min(mins)}")
    print(f"\nKEY CELLS for the truth-vs-support separation:")
    print(f"  correct_low  (weakly-supported correct) -- omitted by Cheang")
    print(f"  incorrect_high (strongly-supported false) -- the confident errors")
    print(f"If both are >= ~30-50 per model, the support-matched truth test is")
    print(f"feasible. If some models have tiny cells, we pool or report subset.")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
