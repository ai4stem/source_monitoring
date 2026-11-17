"""
semantic_entropy_nli_fast.py  (SERVER-SIDE)  —  faster NLI semantic entropy
===========================================================================
Same method as semantic_entropy_nli.py (Farquhar-style entailment clustering)
but rewritten to actually finish:

  * caps facts per model (--n_facts, default 400) -- AUROC is stable at a few
    hundred, so processing all 6000 was the main waste.
  * BATCHES all NLI comparisons for one fact into a single forward pass instead
    of one call per pair (the old one-at-a-time loop on CPU was the bottleneck).
  * forces and verifies GPU placement, prints the device, and prints progress
    every 50 facts so you can see it move.
  * de-duplicates identical answers before clustering (many samples repeat), which
    cuts NLI calls sharply.

Clustering: greedy. For a new answer, compare it (bidirectionally) against the
current cluster representatives IN ONE BATCH. Two answers share a cluster iff each
entails the other, given the question as shared context.

INPUT : outputs/probed_<model>.jsonl  (raw answers)
OUTPUT: outputs/semantic_entropy_nli.json
Usage:
    python semantic_entropy_nli_fast.py --model all --n_facts 400 --verbose
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
    if not stab:
        return [], []
    ans = stab.get("answers"); cor = stab.get("correct")
    if ans:
        return ans, (cor if cor else [])
    a, c = [], []
    for pa in stab.get("by_paraphrase", []):
        a.extend(pa.get("answers", [])); c.extend(pa.get("correct", []))
    return a, c


class NLIBatch:
    def __init__(self, nli_name, device="cuda"):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(nli_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(nli_name)
        self.model.eval()
        if device == "cuda" and not torch.cuda.is_available():
            print("  [warn] cuda not available, falling back to CPU (will be slow)")
            device = "cpu"
        self.device = device
        self.model.to(device)
        print(f"  [nli] model on device: {next(self.model.parameters()).device}")
        labels = self.model.config.id2label
        self.ent_idx = next((i for i, l in labels.items()
                             if "entail" in l.lower()), 2)

    def entails_batch(self, pairs):
        """pairs: list of (premise, hypothesis). Returns list[bool] entailment."""
        if not pairs:
            return []
        prem = [p for p, _ in pairs]; hyp = [h for _, h in pairs]
        enc = self.tok(prem, hyp, return_tensors="pt", truncation=True,
                       max_length=128, padding=True).to(self.device)
        with self.torch.no_grad():
            logits = self.model(**enc).logits
        preds = logits.argmax(dim=1).tolist()
        return [p == self.ent_idx for p in preds]

    def cluster(self, answers, ctx=""):
        """Greedy semantic clustering with batched bidirectional entailment."""
        # dedupe identical strings first
        uniq = []
        dup_count = Counter()
        for a in answers:
            key = a.strip().lower()
            if key not in dup_count:
                uniq.append(a)
            dup_count[key] += 1
        reps = []            # representative answer per cluster
        rep_keys = []        # normalized key of representative
        counts = []
        for a in uniq:
            key = a.strip().lower()
            if reps:
                # bidirectional: a<->rep for every rep, in one batch
                fwd = [(f"{ctx} {a}".strip(), f"{ctx} {r}".strip()) for r in reps]
                bwd = [(f"{ctx} {r}".strip(), f"{ctx} {a}".strip()) for r in reps]
                res = self.entails_batch(fwd + bwd)
                n = len(reps)
                placed = False
                for k in range(n):
                    if res[k] and res[n + k]:
                        counts[k] += dup_count[key]; placed = True; break
                if placed:
                    continue
            reps.append(a); rep_keys.append(key); counts.append(dup_count[key])
        return counts


def semantic_entropy_from_counts(counts):
    tot = sum(counts)
    if tot == 0:
        return 0.0
    return -sum((c / tot) * math.log(c / tot) for c in counts)


def run_model(path, nli, n_facts=400, max_answers=8, verbose=False):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    facts = [r for r in rows if r.get("item_type") == "fact"]
    # keep only facts with usable raw answers, then cap
    usable = []
    for r in facts:
        a, c = get_answers_correct(r.get("stability") or {})
        if a and c:
            usable.append((r, a, c))
    # cap while keeping class balance-ish: just take first n_facts
    usable = usable[:n_facts]
    if len(usable) < 50:
        print(f"  [skip] {os.path.basename(path)}: too few ({len(usable)})")
        return None

    se_scores, y_correct, y_confab = [], [], []
    for idx, (r, answers, correct) in enumerate(usable):
        answers = answers[:max_answers]
        counts = nli.cluster(answers, ctx=r.get("question", ""))
        se_scores.append(semantic_entropy_from_counts(counts))
        y_correct.append(0 if sum(correct) / len(correct) >= 0.5 else 1)
        src = (r.get("claims") or [{}])[0].get("source")
        y_confab.append(1 if src == "confabulated" else (0 if src == "parametric" else None))
        if verbose and (idx + 1) % 50 == 0:
            print(f"    {idx+1}/{len(usable)}")

    key = os.path.basename(path).replace("probed_", "").replace(".jsonl", "")
    auroc_corr = _auroc(se_scores, y_correct)
    pairs = [(s, y) for s, y in zip(se_scores, y_confab) if y is not None]
    auroc_confab = _auroc([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else float("nan")
    res = {"model": key, "n": len(usable),
           "auroc_incorrect_nli": round(auroc_corr, 4),
           "auroc_confabulated_nli": round(auroc_confab, 4) if auroc_confab == auroc_confab else None}
    print(f"  {key}: NLI-SE AUROC (incorrect)={res['auroc_incorrect_nli']}  "
          f"(confab)={res['auroc_confabulated_nli']}  n={res['n']}")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all")
    p.add_argument("--glob", default="outputs/probed_*.jsonl")
    p.add_argument("--nli", default="microsoft/deberta-large-mnli")
    p.add_argument("--out", default="outputs/semantic_entropy_nli.json")
    p.add_argument("--n_facts", type=int, default=400)
    p.add_argument("--max_answers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print(f"[nli] loading {args.nli} ...")
    nli = NLIBatch(args.nli, device=args.device)

    paths = sorted(glob.glob(args.glob)) if args.model == "all" \
        else [f"outputs/probed_{args.model}.jsonl"]
    rows = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[skip] {path} not found"); continue
        print(f"[nli] {os.path.basename(path)}")
        r = run_model(path, nli, n_facts=args.n_facts,
                      max_answers=args.max_answers, verbose=args.verbose)
        if r:
            rows.append(r)
    if not rows:
        print("\nNo results."); return
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)

    import statistics as st
    a = [r["auroc_incorrect_nli"] for r in rows]
    print(f"\n{'model':18s} {'NLI-SE (incorrect)':>18s}")
    print("-" * 38)
    for r in rows:
        print(f"{r['model']:18s} {r['auroc_incorrect_nli']:>18.3f}")
    print("-" * 38)
    print(f"{'MEAN':18s} {st.mean(a):>18.3f}")
    print(f"\ncompare: lightweight SE ~0.77, internal nested ~0.75")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
