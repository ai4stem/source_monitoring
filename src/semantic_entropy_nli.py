"""
semantic_entropy_nli.py  (SERVER-SIDE)  —  proper semantic entropy (Farquhar)
=============================================================================
Motivation: our "lightweight semantic entropy" clusters answers by surface-form
normalisation, not by meaning. Farquhar et al. cluster by bidirectional NLI
entailment. If the proper method scores much higher, the honest comparison may be
"behavioural signals beat internal ones", not "both are weak". This script
computes the NLI-based version so the comparison is fair.

Method (per fact, from the saved probing samples):
  1. take the sampled answers (already stored in stability.answers / by_paraphrase)
  2. cluster them into semantic-equivalence classes: two answers are in the same
     class iff NLI says each ENTAILS the other (bidirectional), given the question
     as context. Greedy: compare each answer to existing cluster representatives.
  3. semantic entropy = Shannon entropy over cluster probabilities (by count).
  4. AUROC of (higher entropy => confabulated/incorrect) vs the label.

We report AUROC for two labels so it slots into either framing:
  * correctness  (correct vs incorrect)      -> matches the 2x2 paper
  * parametric vs confabulated (old label)    -> matches the old table, if present

Uses a standard NLI model (default: microsoft/deberta-large-mnli). Downloadable
from HF; if offline, set --nli to a local path.

INPUT : outputs/probed_<model>.jsonl  (revised probing, has raw answers)
OUTPUT: outputs/semantic_entropy_nli.json
Usage:
    python semantic_entropy_nli.py --model all --verbose
    python semantic_entropy_nli.py --model qwen3-8b --max_answers 10 --verbose
"""
from __future__ import annotations
import argparse, json, os, glob, math, re
from collections import Counter


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


def get_answers_correct(stab):
    """Return (answers, correct_list) from a stability record."""
    if not stab:
        return [], []
    ans = stab.get("answers")
    cor = stab.get("correct")
    if ans:
        return ans, (cor if cor else [])
    a, c = [], []
    for pa in stab.get("by_paraphrase", []):
        a.extend(pa.get("answers", []))
        c.extend(pa.get("correct", []))
    return a, c


class NLICluster:
    def __init__(self, nli_name, device="cuda"):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tok = AutoTokenizer.from_pretrained(nli_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(nli_name)
        self.model.eval()
        self.model.to(device)
        self.device = device
        # find entailment index
        labels = self.model.config.id2label
        self.ent_idx = next((i for i, l in labels.items()
                             if "entail" in l.lower()), 2)

    def entails(self, premise, hypothesis):
        import torch
        with torch.no_grad():
            enc = self.tok(premise, hypothesis, return_tensors="pt",
                           truncation=True, max_length=128).to(self.device)
            logits = self.model(**enc).logits[0]
            pred = int(logits.argmax())
        return pred == self.ent_idx

    def bidirectional(self, a, b, ctx=""):
        pa = f"{ctx} {a}".strip(); pb = f"{ctx} {b}".strip()
        return self.entails(pa, pb) and self.entails(pb, pa)

    def cluster(self, answers, ctx=""):
        """Greedy semantic clustering; returns list of cluster sizes."""
        reps = []  # representative answer per cluster
        counts = []
        for ans in answers:
            placed = False
            for k, rep in enumerate(reps):
                if self.bidirectional(ans, rep, ctx):
                    counts[k] += 1; placed = True; break
            if not placed:
                reps.append(ans); counts.append(1)
        return counts


def semantic_entropy_from_counts(counts):
    tot = sum(counts)
    if tot == 0:
        return 0.0
    return -sum((c / tot) * math.log(c / tot) for c in counts)


def run_model(path, nli, max_answers=10, verbose=False):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    facts = [r for r in rows if r.get("item_type") == "fact"]
    se_scores, y_correct, y_confab = [], [], []
    for r in facts:
        claims = r.get("claims") or [{}]
        src = claims[0].get("source")
        stab = r.get("stability") or {}
        answers, correct = get_answers_correct(stab)
        if not answers or not correct:
            continue
        answers = answers[:max_answers]
        q = r.get("question", "")
        counts = nli.cluster(answers, ctx=q)
        se = semantic_entropy_from_counts(counts)
        # correctness label: majority correct
        is_correct = sum(correct) / len(correct) >= 0.5
        se_scores.append(se)
        y_correct.append(0 if is_correct else 1)   # 1=incorrect (high SE expected)
        if src in ("parametric", "confabulated"):
            y_confab.append(1 if src == "confabulated" else 0)
        else:
            y_confab.append(None)
    if len(se_scores) < 50:
        print(f"  [skip] {os.path.basename(path)}: too few ({len(se_scores)})")
        return None
    key = os.path.basename(path).replace("probed_", "").replace(".jsonl", "")
    auroc_corr = _auroc(se_scores, y_correct)
    # confab AUROC on the subset with a resolved old label
    pairs = [(s, y) for s, y in zip(se_scores, y_confab) if y is not None]
    auroc_confab = _auroc([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else float("nan")
    res = {"model": key, "n": len(se_scores),
           "auroc_incorrect_nli": round(auroc_corr, 4),
           "auroc_confabulated_nli": round(auroc_confab, 4) if auroc_confab == auroc_confab else None}
    if verbose:
        print(f"  {key}: NLI-SE AUROC (incorrect)={res['auroc_incorrect_nli']}  "
              f"(confab)={res['auroc_confabulated_nli']}  n={res['n']}")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all")
    p.add_argument("--glob", default="outputs/probed_*.jsonl")
    p.add_argument("--nli", default="microsoft/deberta-large-mnli")
    p.add_argument("--out", default="outputs/semantic_entropy_nli.json")
    p.add_argument("--max_answers", type=int, default=10,
                   help="cap answers per fact (NLI is O(n*clusters); 10 is plenty)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print(f"[nli] loading NLI model {args.nli} ...")
    nli = NLICluster(args.nli, device=args.device)

    paths = sorted(glob.glob(args.glob)) if args.model == "all" \
        else [f"outputs/probed_{args.model}.jsonl"]
    rows = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[skip] {path} not found"); continue
        print(f"[nli] {os.path.basename(path)}")
        r = run_model(path, nli, max_answers=args.max_answers, verbose=args.verbose)
        if r:
            rows.append(r)
    if not rows:
        print("\nNo results."); return
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)

    import statistics as st
    a = [r["auroc_incorrect_nli"] for r in rows]
    print(f"\n{'model':18s} {'NLI-SE (incorrect)':>18s} {'NLI-SE (confab)':>16s}")
    print("-" * 54)
    for r in rows:
        print(f"{r['model']:18s} {r['auroc_incorrect_nli']:>18.3f} "
              f"{str(r['auroc_confabulated_nli']):>16s}")
    print("-" * 54)
    print(f"{'MEAN':18s} {st.mean(a):>18.3f}")
    print(f"\nCompare this NLI-based semantic entropy to:")
    print(f"  lightweight (surface) semantic entropy: ~0.77 (baseline_uncertainty.json)")
    print(f"  internal nested AUROC:                  ~0.75")
    print(f"  If NLI-SE is much higher (~0.85), behavioural > internal and we say so.")
    print(f"  If similar (~0.77), the 'both weak' conclusion holds under the proper method.")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
