"""
diagnose_conflict.py  (SERVER-SIDE)  —  can we decompose override into ability?
==============================================================================
R7 currently measures DISPOSITION -- how
often the model follows context over its own knowledge under conflict. To also
measure ABILITY -- does the model follow context when context is RIGHT and reject
it when context is WRONG -- we need conflict items where we know whether the
context-asserted answer is actually correct.

This script inspects eval_<model>.jsonl conflict items and reports what is
available, WITHOUT running the model, so we know if Contribution 6 is feasible:
  * fields present on conflict items (target, parametric_lure, answer_aliases,
    gold, is_context_correct, ...)
  * whether any item carries a gold/true answer distinct from the context target
  * how many items would fall into (context-correct) vs (context-wrong) if we can
    tell them apart

If conflict items only ever assert a FALSE answer (context always wrong), we can
only measure rejection ability, not full ability; the script says so. If a gold
field exists, we can split and Contribution 6 is feasible by re-analysis /
a light generation pass.

INPUT : outputs/eval_<model>.jsonl
OUTPUT: prints a report; writes outputs/conflict_fields.json
Usage:
    python diagnose_conflict.py --model qwen3-8b
    python diagnose_conflict.py --model all
"""
from __future__ import annotations
import argparse, json, os, glob


def inspect(path, verbose=False):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    conflict = [r for r in rows if r.get("probe_kind") == "conflict"]
    control = [r for r in rows if r.get("probe_kind") == "non_conflict_control"]
    if not conflict:
        return None
    # what fields exist?
    keys = set()
    for r in conflict[:50]:
        keys.update(r.keys())
    sample = conflict[0]
    # heuristics: does a gold answer distinct from target exist?
    gold_fields = [k for k in ("gold", "answer", "true_answer", "answer_aliases",
                               "correct_answer", "is_context_correct")
                   if k in keys]
    # count how many have parametric_lure distinct from target
    has_lure = sum(1 for r in conflict if r.get("parametric_lure"))
    key = os.path.basename(path).replace("eval_", "").replace(".jsonl", "")
    rep = {
        "model": key, "n_conflict": len(conflict), "n_control": len(control),
        "conflict_fields": sorted(keys),
        "gold_like_fields": gold_fields,
        "has_parametric_lure": has_lure,
        "sample_item": {k: (str(sample.get(k))[:80]) for k in
                        ("question", "context", "target", "parametric_lure",
                         "answer_aliases", "probe_kind") if k in sample},
    }
    if verbose:
        print(f"\n=== {key} ===")
        print(f"  conflict items: {len(conflict)}, control: {len(control)}")
        print(f"  fields on conflict items: {sorted(keys)}")
        print(f"  gold-like fields (for ability split): {gold_fields or 'NONE'}")
        print(f"  items with parametric_lure: {has_lure}/{len(conflict)}")
        print(f"  sample item:")
        for k, v in rep["sample_item"].items():
            print(f"    {k}: {v}")
    return rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all")
    p.add_argument("--glob", default="outputs/eval_*.jsonl")
    p.add_argument("--out", default="outputs/conflict_fields.json")
    args = p.parse_args()
    paths = sorted(glob.glob(args.glob)) if args.model == "all" \
        else [f"outputs/eval_{args.model}.jsonl"]

    reps = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[skip] {path} not found"); continue
        r = inspect(path, verbose=True)
        if r:
            reps.append(r)
    if not reps:
        print("\nNo conflict items found in any eval set."); return
    with open(args.out, "w") as f:
        json.dump(reps, f, indent=2)

    print(f"\n{'='*66}\nFEASIBILITY OF CONTRIBUTION 6 (ability decomposition)\n{'='*66}")
    any_gold = any(r["gold_like_fields"] for r in reps)
    if any_gold:
        print("A gold-like field exists on conflict items. We can split items into")
        print("context-correct vs context-wrong and measure whether the model")
        print("follows correct context and rejects wrong context (ABILITY), not just")
        print("its overall follow rate (DISPOSITION). Contribution 6 is FEASIBLE by")
        print("re-analysis (+ possibly one generation pass to record per-item follow).")
        print(f"gold-like fields seen: {sorted(set(f for r in reps for f in r['gold_like_fields']))}")
    else:
        print("No gold/true-answer field distinct from the context target was found.")
        print("If ConflictQA here always asserts a FALSE answer (context wrong by")
        print("construction), we can only measure REJECTION ability, not the full")
        print("correct-follow/wrong-reject ability. In that case Contribution 6 is")
        print("LIMITED: report override as disposition (current R7) and note that")
        print("context is wrong by design, so low override = appropriate rejection.")
        print("To do the full split we would need conflict items with a KNOWN-correct")
        print("context variant, i.e. a new data construction (a real new experiment).")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
