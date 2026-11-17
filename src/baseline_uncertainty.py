"""
baseline_uncertainty.py  (SERVER-SIDE)  —  EXPERIMENT 2
=======================================================
External behavioural baselines for detecting confabulation, computed from the
RAW probing samples saved by the revised probe_knowledge.py. They provide the
comparison of the internal read-out with standard behavioural uncertainty signals.

Two baselines, both per-fact, using only the closed-book probing samples:

  1. self-consistency:  1 - (fraction of samples equal to the modal answer).
     Low consistency  => likely confabulation. This is the cheap, standard
     sampling-agreement signal.

  2. semantic entropy (lightweight):  cluster the sampled answers by normalized
     string equality (a practical proxy for meaning clusters on short factual
     answers) and take the Shannon entropy over cluster probabilities. Higher
     entropy => likely confabulation. This mirrors Farquhar et al. (2024) at the
     level their method reduces to for short factual answers.

We then ask how well each baseline separates confabulated from parametric facts,
using AUROC (threshold-free, imbalance-robust). If a behavioural baseline achieves high AUROC on the SAME items where the
hidden-state probe is weak (~0.70 acc), that sharpens the paper: the failure is
specific to internal representations, not to detectability in general.

INPUT : outputs/probed_<model>.jsonl   (must be produced by the REVISED probing,
        i.e. each fact's stability record contains 'by_paraphrase' or 'answers'+
        'correct'.)
OUTPUT: outputs/baseline_uncertainty.json

Usage:
    python baseline_uncertainty.py --model all
    python baseline_uncertainty.py --model qwen3-8b --verbose
"""
from __future__ import annotations
import argparse, json, os, glob, math, re
from collections import Counter


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def self_inconsistency(answers: list[str]) -> float:
    """1 - fraction matching the modal normalized answer. In [0,1]; higher=less consistent."""
    if not answers:
        return 0.0
    norm = [_norm(a) for a in answers]
    modal_count = Counter(norm).most_common(1)[0][1]
    return 1.0 - modal_count / len(norm)


def semantic_entropy(answers: list[str]) -> float:
    """Shannon entropy (nats) over normalized-answer clusters. Higher=more spread."""
    if not answers:
        return 0.0
    norm = [_norm(a) for a in answers]
    counts = Counter(norm)
    total = len(norm)
    return -sum((c / total) * math.log(c / total) for c in counts.values())


def _auroc(scores: list[float], labels: list[int]) -> float:
    """AUROC via rank statistic (label 1 = confabulated = positive).
    No sklearn dependency. Handles ties by average rank (Mann-Whitney U)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    # rank all scores, average ranks for ties
    paired = sorted(zip(scores, range(len(scores))), key=lambda x: x[0])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][0] == paired[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[paired[k][1]] = avg
        i = j + 1
    sum_pos = sum(r for r, y in zip(ranks, labels) if y == 1)
    n_pos, n_neg = len(pos), len(neg)
    u = sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _get_answers(stab: dict) -> list[str]:
    """Extract the list of sampled answers from a stability record (revised format)."""
    if not stab:
        return []
    if "answers" in stab and stab["answers"]:
        return stab["answers"]
    if "by_paraphrase" in stab:
        out = []
        for pa in stab["by_paraphrase"]:
            out.extend(pa.get("answers", []))
        return out
    return []


def run_model(path: str, verbose=False) -> dict | None:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    facts = [r for r in rows if r.get("item_type") == "fact"]
    sc_scores, se_scores, labels = [], [], []
    n_missing_raw = 0
    for r in facts:
        claims = r.get("claims") or []
        if not claims:
            continue
        src = claims[0].get("source")
        if src not in ("parametric", "confabulated"):
            continue  # skip grey/verbatim for the main parametric-vs-confab AUROC
        stab = r.get("stability") or {}
        answers = _get_answers(stab)
        if not answers:
            n_missing_raw += 1
            continue
        y = 1 if src == "confabulated" else 0
        sc_scores.append(self_inconsistency(answers))
        se_scores.append(semantic_entropy(answers))
        labels.append(y)
    if len(set(labels)) < 2:
        print(f"  [skip] {os.path.basename(path)}: not both classes present "
              f"(missing_raw={n_missing_raw})")
        return None
    key = os.path.basename(path).replace("probed_", "").replace(".jsonl", "")
    res = {
        "model": key,
        "n_parametric": int(labels.count(0)),
        "n_confabulated": int(labels.count(1)),
        "auroc_self_consistency": round(_auroc(sc_scores, labels), 4),
        "auroc_semantic_entropy": round(_auroc(se_scores, labels), 4),
        "n_missing_raw": n_missing_raw,
    }
    if verbose:
        print(f"  {key}: SC AUROC={res['auroc_self_consistency']} "
              f"SE AUROC={res['auroc_semantic_entropy']} "
              f"(n_param={res['n_parametric']}, n_confab={res['n_confabulated']}, "
              f"missing_raw={n_missing_raw})")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all")
    p.add_argument("--glob", default="outputs/probed_*.jsonl")
    p.add_argument("--out", default="outputs/baseline_uncertainty.json")
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
        print("\nNo results. Did you re-run probing with the REVISED "
              "probe_knowledge.py? The raw per-sample answers must be saved in "
              "each fact's 'stability' record.")
        return

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)

    # summary
    import statistics as st
    sc = [r["auroc_self_consistency"] for r in rows]
    se = [r["auroc_semantic_entropy"] for r in rows]
    print(f"\n{'model':18s} {'SC_AUROC':>9s} {'SE_AUROC':>9s} {'n_param':>8s} {'n_confab':>9s}")
    print("-" * 58)
    for r in rows:
        print(f"{r['model']:18s} {r['auroc_self_consistency']:>9.3f} "
              f"{r['auroc_semantic_entropy']:>9.3f} {r['n_parametric']:>8d} "
              f"{r['n_confabulated']:>9d}")
    print("-" * 58)
    print(f"{'MEAN':18s} {st.mean(sc):>9.3f} {st.mean(se):>9.3f}")
    print(f"\nCompare to the hidden-state probe (~0.70 accuracy, AUROC to be added).")
    print(f"If these behavioural AUROCs are substantially higher, the paper's claim "
          f"sharpens: behavioural signals detect confabulation that the internal "
          f"representation does not expose.")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
