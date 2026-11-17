"""
recompute_support_semantic.py  —  meaning-level LOPO support from the stored answers
=====================================================================================
WHY. extract_hidden_2x2.py scores agreement between paraphrases by exact match of the
normalised answer STRING. Two of the six prompt variants carry no brevity instruction
and elicit paragraph-length answers, so six responses that all give the same answer
often count as six different answers (support 0.2-0.33). That measures surface-form
agreement, not whether the model gives the same answer.

WHAT. This script re-clusters the six stored answers of every fact by the answer they
give, then recomputes the same leave-one-paraphrase-out support on the cluster labels.
No model is run: only outputs/probed_<model>.jsonl is read. Hidden states are untouched.

Two clusterings are written so the choice can be checked rather than trusted:

  anchor  (primary; correctness-blind)
      Each answer is reduced to a short canonical form: the first **bold** span if the
      answer has one, else the whole answer if it is <= MAX_SHORT words; words that
      only repeat the question are dropped. Canonical
      forms are merged when equal or when one contains the other as a whole-word
      phrase ("spinoza" in "baruch spinoza"). A long answer with no canonical form is
      attached to the cluster whose form appears earliest in it; otherwise it stays a
      singleton. Gold answers are never consulted.
  gold    (sensitivity)
      As above, but every answer already scored correct (alias match) is first placed
      in one shared cluster.

OUTPUT  outputs/support_semantic_<model>.npz
    support_string   float32 [n]  recomputed string support (must equal hidden2x2 file)
    support_anchor   float32 [n]
    support_gold     float32 [n]
    entropy_anchor   float32 [n]  Shannon entropy (nats) over anchor clusters
    entropy_gold     float32 [n]
    correct          int8    [n]
    frac_singleton_long float     share of long answers left unattached (diagnostic)
Row order is identical to outputs/hidden2x2_<model>.npz (asserted).

Usage:
    python recompute_support_semantic.py --model all
"""
from __future__ import annotations
import argparse, glob, json, math, os, re
from collections import Counter
import numpy as np

MAX_SHORT = 8          # words; longer answers need a bold span or an anchor to be clustered
MIN_ANCHOR_CHARS = 3   # one- or two-character forms only match exactly


def _norm(s: str) -> str:                       # identical to extract_hidden_2x2._norm
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _strip_question(form: str, qwords) -> str:
    """Drop words that merely repeat the question, so a form is the answer content only."""
    kept = [w for w in form.split() if w not in qwords]
    return " ".join(kept)


def canonical(raw: str, qwords=frozenset()):
    """Short canonical form of an answer, or None if it is a long unmarked answer
    (or says nothing beyond the question's own words)."""
    m = re.search(r"\*\*(.+?)\*\*", raw, flags=re.S)
    if m:
        c = _norm(m.group(1))
        if c and len(c.split()) <= MAX_SHORT:
            c = _strip_question(c, qwords)
            if c:
                return c
    n = _norm(raw)
    if n and len(n.split()) <= MAX_SHORT:
        return _strip_question(n, qwords) or None
    return None


def _contains(long: str, short: str) -> bool:
    if len(short) < MIN_ANCHOR_CHARS:
        return False
    return re.search(r"(?<!\w)" + re.escape(short) + r"(?!\w)", long) is not None


def cluster(raws, correct=None, question=""):
    """Cluster labels for the answers of one fact. correct=None -> correctness-blind."""
    n = len(raws)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    norms = [_norm(r) for r in raws]
    qwords = frozenset(_norm(question).split())
    canon = [canonical(r, qwords) for r in raws]
    locked = [False] * n
    if correct is not None:                      # gold-assisted: correct answers share a cluster
        cor = [i for i in range(n) if correct[i]]
        for i in cor[1:]:
            union(cor[0], i)
        for i in cor:
            locked[i] = True
    free = [i for i in range(n) if not locked[i]]
    short = [i for i in free if canon[i] is not None]
    for a in range(len(short)):
        for b in range(a + 1, len(short)):
            i, j = short[a], short[b]
            if canon[i] == canon[j] or _contains(canon[i], canon[j]) or _contains(canon[j], canon[i]):
                union(i, j)
    n_long = n_single = 0
    for i in free:
        if canon[i] is not None:
            continue
        n_long += 1
        best, pos = None, None
        for j in short:
            if len(canon[j]) < MIN_ANCHOR_CHARS:
                continue
            m = re.search(r"(?<!\w)" + re.escape(canon[j]) + r"(?!\w)", norms[i])
            if m and (pos is None or m.start() < pos):
                best, pos = j, m.start()
        if best is not None:
            union(i, best)
        else:
            same = [j for j in free if j != i and canon[j] is None and norms[j] == norms[i]]
            if same:
                union(i, same[0])
            else:
                n_single += 1
    return [find(i) for i in range(n)], n_long, n_single


def lopo(labels) -> float:
    n = len(labels); sups = []
    for i in range(n):
        others = [labels[j] for j in range(n) if j != i]
        sups.append(Counter(others).most_common(1)[0][1] / len(others))
    return float(np.mean(sups))


def entropy(labels) -> float:
    c = Counter(labels); tot = len(labels)
    return -sum((v / tot) * math.log(v / tot) for v in c.values())


def run(key):
    path = f"outputs/probed_{key}.jsonl"
    ref_path = f"outputs/hidden2x2_{key}.npz"
    if not (os.path.exists(path) and os.path.exists(ref_path)):
        print(f"[skip] {key}"); return None
    s_str, s_anc, s_gold, e_anc, e_gold, cor = [], [], [], [], [], []
    tot_long = tot_single = 0
    for line in open(path):
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
        # string support exactly as extract_hidden_2x2 (paraphrase-level LOPO)
        paras = [[_norm(a) for a in pa.get("answers", [])] for pa in bp]
        sups = []
        for i in range(len(paras)):
            others = [a for j, p in enumerate(paras) if j != i for a in p]
            if others:
                sups.append(Counter(others).most_common(1)[0][1] / len(others))
        s_str.append(float(np.mean(sups)))
        q = x.get("question") or ""
        la, nl, ns = cluster(raws, question=q)
        lg, _, _ = cluster(raws, correct=flags, question=q)
        tot_long += nl; tot_single += ns
        s_anc.append(lopo(la)); s_gold.append(lopo(lg))
        e_anc.append(entropy(la)); e_gold.append(entropy(lg))
        cor.append(1 if sum(flags) / len(flags) >= 0.5 else 0)
    ref = np.load(ref_path)
    n = len(ref["support"])
    s_str = np.array(s_str[:n], dtype=np.float32)
    assert len(s_str) == n, (key, len(s_str), n)
    assert np.allclose(s_str, ref["support"], atol=1e-5), f"{key}: row order differs from {ref_path}"
    assert np.array_equal(np.array(cor[:n], dtype=np.int8), ref["correct"]), f"{key}: correctness differs"
    out = dict(support_string=s_str,
               support_anchor=np.array(s_anc[:n], dtype=np.float32),
               support_gold=np.array(s_gold[:n], dtype=np.float32),
               entropy_anchor=np.array(e_anc[:n], dtype=np.float32),
               entropy_gold=np.array(e_gold[:n], dtype=np.float32),
               correct=ref["correct"],
               frac_singleton_long=np.float32(tot_single / max(tot_long, 1)))
    np.savez_compressed(f"outputs/support_semantic_{key}.npz", **out)
    c = ref["correct"] == 1
    print(f"{key:17s} n={n}  string {s_str.mean():.2f} | anchor {out['support_anchor'].mean():.2f} "
          f"(correct {out['support_anchor'][c].mean():.2f}, incorrect {out['support_anchor'][~c].mean():.2f}) | "
          f"gold {out['support_gold'].mean():.2f} | long answers left singleton {tot_single}/{tot_long} "
          f"({tot_single / max(tot_long, 1):.0%})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all")
    args = ap.parse_args()
    keys = sorted(os.path.basename(p)[10:-4] for p in glob.glob("outputs/hidden2x2_*.npz")) \
        if args.model == "all" else [args.model]
    for k in keys:
        run(k)


if __name__ == "__main__":
    main()
