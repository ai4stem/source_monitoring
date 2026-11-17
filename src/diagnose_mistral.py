"""
diagnose_mistral.py  (SERVER-SIDE)
==================================
Why is mistral-7b an outlier (baseline AUROC 0.58 vs ~0.79 for others; and its
self-consistency AUROC exactly equals its semantic-entropy AUROC, which happens
for no other model)?

Hypotheses to check from the raw probed samples:
  H1. Degenerate generations: mistral-7b emits empty / repeated / boilerplate
      answers, so 'answers' are all-identical or all-different regardless of
      whether the fact is known -> both SC and SE become the same near-useless
      signal.
  H2. Formatting artefact: answers carry chat scaffolding / refusals that the
      normalizer doesn't strip, collapsing the answer space.
  H3. Genuinely flat confidence: mistral really does sample the same way for
      known and unknown facts.

This script reads outputs/probed_mistral-7b.jsonl and, for a sample of parametric
and confabulated facts, prints the raw answers and simple statistics:
  - mean unique-answer count per fact (parametric vs confabulated)
  - fraction of facts whose answers are ALL identical / ALL distinct
  - fraction empty or very short
  - a few concrete examples
Compare against a healthy model (default qwen3-8b) to see the contrast.

Usage:
    python diagnose_mistral.py
    python diagnose_mistral.py --model mistral-7b --ref qwen3-8b --n 12
"""
from __future__ import annotations
import argparse, json, re
from collections import Counter


def _norm(s):
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def get_answers(stab):
    if not stab:
        return []
    if stab.get("answers"):
        return stab["answers"]
    out = []
    for pa in stab.get("by_paraphrase", []):
        out.extend(pa.get("answers", []))
    return out


def stats_for(path, label_want, n_show=6):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    facts = [r for r in rows if r.get("item_type") == "fact"]
    sel = []
    for r in facts:
        c = (r.get("claims") or [{}])[0]
        if c.get("source") == label_want:
            sel.append(r)
    uniq_counts, all_same, all_diff, empties, shorts = [], 0, 0, 0, 0
    examples = []
    for r in sel:
        ans = get_answers(r.get("stability") or {})
        if not ans:
            continue
        norm = [_norm(a) for a in ans]
        u = len(set(norm))
        uniq_counts.append(u)
        if u == 1:
            all_same += 1
        if u == len(norm):
            all_diff += 1
        empties += sum(1 for a in norm if a == "")
        shorts += sum(1 for a in norm if 0 < len(a) <= 2)
        if len(examples) < n_show:
            examples.append((r.get("question", "")[:50], ans[:4]))
    import statistics as st
    n = len(uniq_counts)
    return {
        "label": label_want, "n_facts": n,
        "mean_unique": round(st.mean(uniq_counts), 2) if n else None,
        "frac_all_same": round(all_same / n, 3) if n else None,
        "frac_all_distinct": round(all_diff / n, 3) if n else None,
        "empty_answers": empties, "short_answers": shorts,
        "examples": examples,
    }


def report(path, model):
    print(f"\n{'='*70}\n{model}   ({path})\n{'='*70}")
    for lab in ("parametric", "confabulated"):
        s = stats_for(path, lab)
        if s["n_facts"] == 0:
            print(f"  [{lab}] none found"); continue
        print(f"  [{lab}] n={s['n_facts']}  mean_unique_answers={s['mean_unique']}  "
              f"all_same={s['frac_all_same']}  all_distinct={s['frac_all_distinct']}  "
              f"empty={s['empty_answers']} short={s['short_answers']}")
        for q, ans in s["examples"][:4]:
            shown = " | ".join(a[:25] for a in ans)
            print(f"      Q: {q:50s}  A: {shown}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mistral-7b")
    p.add_argument("--ref", default="qwen3-8b")
    p.add_argument("--n", type=int, default=6)
    args = p.parse_args()

    report(f"outputs/probed_{args.model}.jsonl", args.model)
    report(f"outputs/probed_{args.ref}.jsonl", args.ref)

    print(f"\n{'='*70}\nWHAT TO LOOK FOR\n{'='*70}")
    print("If mistral shows mean_unique ~1 for BOTH parametric and confabulated")
    print("(answers always identical), or all_distinct ~1 for both (always different),")
    print("then SC and SE carry the same degenerate signal -> that explains the")
    print("0.58 AUROC and SC==SE. Compare with the reference model, which should")
    print("show mean_unique LOW for parametric and HIGHER for confabulated.")
    print("Also check the printed answers for empty strings, refusals, or repeated")
    print("boilerplate that the normaliser cannot separate.")


if __name__ == "__main__":
    main()
