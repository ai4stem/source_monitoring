"""
nli_cluster_support.py  —  entailment-based answer clustering for ALL facts (GPU)
=====================================================================================
WHY. recompute_support_semantic.py clusters the six stored answers of a fact with
rules (canonical short form + containment). That leaves many paragraph-length answers
unclustered. This script is the independent check: answers are clustered by
bidirectional entailment (Farquhar et al. 2024) with DeBERTa-large-MNLI, for every
fact of every model, and the same leave-one-prompt-out support is recomputed.

No LLM is run. Only outputs/probed_<model>.jsonl is read; hidden states are untouched.

For every ordered pair of distinct answers to the same question the 3-way NLI label
is stored, so every clustering below is derived offline from the same predictions.

  text fed to the NLI model
    raw     "<question> <answer>"                       (answer exactly as generated)
    canon   "<question> <short form>"  where the rule-based canonical short form
            exists (first **bold** span, or the answer if <= 8 words), else the raw
            answer. Correctness-blind. Removes the elaboration that makes two long
            answers giving the same answer fail mutual entailment.
  merge criterion
    strict  both directions predicted "entailment"
    loose   neither direction "contradiction" and not both "neutral"
            (the default of Farquhar et al.'s released code)
  clustering  greedy: an answer joins the first cluster whose first member it matches.

OUTPUT  outputs/support_nli_<model>.npz
    support_<text>_<crit>, entropy_<text>_<crit>   float32 [n]   (4 combinations)
    correct int8 [n];  n_pairs_scored;  diagnostics (see _diagnostics)
Row order identical to outputs/hidden2x2_<model>.npz (asserted).

Usage:
    python nli_cluster_support.py --model all --gpus 0,1,2,3
    python nli_cluster_support.py --model qwen3_8b --device cuda:0 --limit 300   # test
"""
from __future__ import annotations
import argparse, glob, json, os, subprocess, sys, time
import numpy as np
from recompute_support_semantic import _norm, canonical, lopo, entropy

NLI = "microsoft/deberta-large-mnli"
MAX_LEN = 320


def load_facts(key):
    """Same filter and order as recompute_support_semantic.run()."""
    facts = []
    for line in open(f"outputs/probed_{key}.jsonl"):
        x = json.loads(line)
        if x.get("item_type") != "fact":
            continue
        bp = (x.get("stability") or {}).get("by_paraphrase")
        if not bp:
            continue
        raws, flags = [], []
        for pa in bp:
            raws.extend(pa.get("answers", [])); flags.extend(pa.get("correct", []))
        if len(bp) < 2 or not flags:
            continue
        facts.append((x.get("question") or "", raws, flags))
    return facts


def texts_for(q, raws, mode):
    if mode == "raw":
        return [" ".join(r.split()) for r in raws]
    qwords = frozenset(_norm(q).split())
    out = []
    for r in raws:
        c = canonical(r, qwords)
        out.append(c if c else " ".join(r.split()))
    return out


class Scorer:
    def __init__(self, device, fp16=False):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(NLI)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI).to(device).eval()
        self.device = device; self.fp16 = fp16
        lab = {i: l.lower() for i, l in self.model.config.id2label.items()}
        self.ent = next(i for i, l in lab.items() if "entail" in l)
        self.con = next(i for i, l in lab.items() if "contra" in l)

    def predict(self, pairs, bs=256):
        """pairs: list of (premise, hypothesis) -> np.int8 labels (0 con, 1 neu, 2 ent)."""
        torch = self.torch
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        out = np.zeros(len(pairs), dtype=np.int8); n_trunc = 0
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = self.tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], return_tensors="pt",
                           truncation=True, max_length=MAX_LEN, padding=True)
            if enc["input_ids"].shape[1] >= MAX_LEN:
                n_trunc += int((enc["attention_mask"].sum(1) >= MAX_LEN).sum())
            enc = enc.to(self.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                p = self.model(**enc).logits.argmax(1).cpu().numpy()
            out[idx] = np.where(p == self.ent, 2, np.where(p == self.con, 0, 1))
        return out, n_trunc


def greedy(texts, lab, crit):
    """lab[(a, b)] = label of premise a, hypothesis b. Returns cluster labels."""
    reps, labels = [], []
    for t in texts:
        k = None
        for ci, r in enumerate(reps):
            if t == r:
                k = ci; break
            f, b = lab[(t, r)], lab[(r, t)]
            ok = (f == 2 and b == 2) if crit == "strict" else (f != 0 and b != 0 and not (f == 1 and b == 1))
            if ok:
                k = ci; break
        if k is None:
            reps.append(t); k = len(reps) - 1
        labels.append(k)
    return labels


def _diagnostics(facts, all_labels):
    """Share of same-fact answer pairs placed in one cluster, by correctness of the pair."""
    cc = [0, 0]; ci = [0, 0]
    for (_, _, flags), labels in zip(facts, all_labels):
        n = len(labels)
        for i in range(n):
            for j in range(i + 1, n):
                same = labels[i] == labels[j]
                if flags[i] and flags[j]:
                    cc[0] += same; cc[1] += 1
                elif bool(flags[i]) != bool(flags[j]):
                    ci[0] += same; ci[1] += 1
    return cc[0] / max(cc[1], 1), ci[0] / max(ci[1], 1)


def run(key, device, limit=None, fp16=False, check_fp32=False):
    t0 = time.time()
    facts = load_facts(key)
    ref = np.load(f"outputs/hidden2x2_{key}.npz")
    n = len(ref["support"]); facts = facts[:n]
    cor = np.array([1 if sum(f) / len(f) >= 0.5 else 0 for _, _, f in facts], dtype=np.int8)
    assert len(facts) == n and np.array_equal(cor, ref["correct"]), f"{key}: rows differ from hidden2x2"
    if limit:
        facts = facts[:limit]; cor = cor[:limit]
    sc = Scorer(device, fp16)
    out = {"correct": cor}; diag = {}
    for mode in ("raw", "canon"):
        per_fact = []; need = {}
        for q, raws, _ in facts:
            ts = texts_for(q, raws, mode); per_fact.append((q, ts))
            u = list(dict.fromkeys(ts))
            for a in u:
                for b in u:
                    if a != b:
                        need[(q, a, b)] = None
        keys = list(need)
        preds, n_trunc = sc.predict([(f"{q} {a}".strip(), f"{q} {b}".strip()) for q, a, b in keys])
        if check_fp32 and mode == "raw":
            sc.fp16 = False; p32, _ = sc.predict([(f"{q} {a}".strip(), f"{q} {b}".strip()) for q, a, b in keys]); sc.fp16 = fp16
            print(f"  fp16 vs fp32 label agreement: {(p32 == preds).mean():.4f} on {len(keys)} pairs")
        for k, p in zip(keys, preds):
            need[k] = int(p)
        diag[f"n_pairs_{mode}"] = len(keys); diag[f"n_truncated_{mode}"] = n_trunc
        for crit in ("strict", "loose"):
            sup, ent, labs = [], [], []
            for q, ts in per_fact:
                lab = {(a, b): need[(q, a, b)] for a in set(ts) for b in set(ts) if a != b}
                l = greedy(ts, lab, crit); labs.append(l)
                sup.append(lopo(l)); ent.append(entropy(l))
            out[f"support_{mode}_{crit}"] = np.array(sup, dtype=np.float32)
            out[f"entropy_{mode}_{crit}"] = np.array(ent, dtype=np.float32)
            a, b = _diagnostics(facts, labs)
            diag[f"same_cluster_both_correct_{mode}_{crit}"] = a
            diag[f"same_cluster_correct_incorrect_{mode}_{crit}"] = b
    for k, v in diag.items():
        out[k] = np.float32(v)
    if not limit:
        np.savez_compressed(f"outputs/support_nli_{key}.npz", **out)
    c = cor == 1
    print(f"[done] {key} n={len(cor)} {time.time() - t0:.0f}s  pairs raw {diag['n_pairs_raw']} "
          f"(trunc {diag['n_truncated_raw']}) canon {diag['n_pairs_canon']}")
    for mode in ("raw", "canon"):
        for crit in ("strict", "loose"):
            s = out[f"support_{mode}_{crit}"]
            print(f"   {mode:5s} {crit:6s} support {s.mean():.3f} (correct {s[c].mean():.3f}, incorrect {s[~c].mean():.3f}) "
                  f"unanimous {np.mean(s >= 0.999):.3f} | same-cluster: both-correct "
                  f"{diag[f'same_cluster_both_correct_{mode}_{crit}']:.3f}, correct-incorrect "
                  f"{diag[f'same_cluster_correct_incorrect_{mode}_{crit}']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gpus", default=None, help="e.g. 0,1,2,3 : one worker process per GPU")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--check_fp32", action="store_true")
    args = ap.parse_args()
    keys = sorted(os.path.basename(p)[10:-4] for p in glob.glob("outputs/hidden2x2_*.npz")) \
        if args.model == "all" else args.model.split(",")
    if args.gpus:
        gpus = args.gpus.split(",")
        sizes = {k: os.path.getsize(f"outputs/probed_{k}.jsonl") for k in keys}
        keys = sorted(keys, key=lambda k: -sizes[k])
        procs = []
        for gi, g in enumerate(gpus):
            mine = keys[gi::len(gpus)]
            if mine:
                procs.append(subprocess.Popen([sys.executable, __file__, "--model", ",".join(mine),
                                               "--device", f"cuda:{g}"]))
        sys.exit(max(p.wait() for p in procs))
    for k in keys:
        run(k, args.device, args.limit, check_fp32=args.check_fp32)


if __name__ == "__main__":
    main()
