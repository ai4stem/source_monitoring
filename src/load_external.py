"""
load_external.py  (SERVER-SIDE: needs `datasets` + internet)
============================================================
Auto-downloads large public QA datasets from HuggingFace and converts them into
our Instance schema, oversampling so probing can filter down to clean labels.

Sources (all Apache/MIT-style, citable):
  - TriviaQA         -> fact candidates likely PARAMETRIC (easy, memorized facts)
  - PopQA            -> fact candidates likely CONFABULATED (long-tail entities)
  - osunlp/ConflictQA-> CONFLICT items (counter_memory as verbatim context,
                        ground_truth as the parametric temptation) [PopQA-based]
  - Natural Questions-> OOD evaluation pool (NOT used in training)

Design rule preserved: fact source labels stay 'pending' here; the model-relative
stability probing (probe_knowledge) assigns parametric / confabulated / gray-zone.
Conflict items are verbatim by construction (kappa=1).

`datasets` is imported lazily so this module loads without it. On a machine
without internet, supply local paths via the `data_dir` args.
"""

from __future__ import annotations
from schema import Instance, Claim
import random


def _hf_load(name, config=None, split="train", data_dir=None):
    """Load a HF dataset. datasets>=4.0 removed script-based loading, so for
    repos that still ship a loading script (e.g. osunlp/ConflictQA) we fall back
    to the auto-generated Parquet mirror at revision 'refs/convert/parquet'."""
    from datasets import load_dataset

    def _try(**extra):
        if data_dir:
            return load_dataset(name, config, split=split, data_dir=data_dir, **extra)
        if config:
            return load_dataset(name, config, split=split, **extra)
        return load_dataset(name, split=split, **extra)

    try:
        return _try()
    except RuntimeError as e:
        if "scripts are no longer supported" in str(e) or "loading script" in str(e):
            # retry against the parquet-converted branch HF maintains automatically
            return _try(revision="refs/convert/parquet")
        raise


# ---------------------------------------------------------------------------
# SQuAD -> easy well-known facts; strong PARAMETRIC source after probing.
# SQuAD answers are short spans and questions cover popular topics, so an 8B
# model reliably knows many of them closed-book (probing keeps only those).
# ---------------------------------------------------------------------------
def load_squad_facts(n=4000, seed=5, split="train") -> list[Instance]:
    ds = _hf_load("rajpurkar/squad", split=split)
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out, used, seen_q = [], 0, set()
    for i in idx:
        if used >= n:
            break
        ex = ds[i]
        q = (ex.get("question") or "").strip()
        ans_texts = (ex.get("answers") or {}).get("text") or []
        if not q or not ans_texts:
            continue
        # dedupe identical questions; skip very context-dependent phrasings
        if q.lower() in seen_q:
            continue
        low = q.lower()
        if low.startswith(("according to", "in the passage", "based on")):
            continue
        seen_q.add(q.lower())
        aliases = [a.strip() for a in ans_texts if a.strip()]
        out.append(Instance(
            id=f"squad_{i}", item_type="fact", probe_kind="parametric_fact",
            context="", question=q, target=aliases[0],
            answer_aliases=aliases,
            claims=[Claim(text=aliases[0], source="pending", conflict=0, labeled=True)],
        ))
        used += 1
    return out


# ---------------------------------------------------------------------------
# TriviaQA -> parametric-leaning fact candidates
# ---------------------------------------------------------------------------
def load_triviaqa_facts(n=3000, seed=0, split="train") -> list[Instance]:
    """Easy factual QA -> most should probe as PARAMETRIC. Oversample; probing
    will demote unstable ones to gray-zone/confabulated."""
    ds = _hf_load("mandarjoshi/trivia_qa", config="rc.nocontext", split=split)
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out, used = [], 0
    for i in idx:
        if used >= n:
            break
        ex = ds[i]
        q = ex["question"].strip()
        ans_field = ex["answer"] if isinstance(ex["answer"], dict) else {}
        ans = (ans_field.get("value") or "").strip() if ans_field else str(ex["answer"]).strip()
        # TriviaQA provides aliases + normalized_aliases -> use for lenient match
        aliases = []
        if ans_field:
            aliases = list(ans_field.get("aliases") or []) + \
                      list(ans_field.get("normalized_aliases") or [])
        if ans:
            aliases = [ans] + aliases
        if not q or not ans:
            continue
        out.append(Instance(
            id=f"trivia_{i}", item_type="fact", probe_kind="parametric_fact",
            context="", question=q, target=ans,
            answer_aliases=aliases or [ans],
            claims=[Claim(text=ans, source="pending", conflict=0, labeled=True)],
        ))
        used += 1
    return out


# ---------------------------------------------------------------------------
# PopQA -> long-tail; oversample to harvest CONFABULATED after probing
# ---------------------------------------------------------------------------
def load_popqa_facts(n=3000, seed=1, split="test") -> list[Instance]:
    """Long-tail entity facts -> many should probe as CONFABULATED (model doesn't
    reliably know them). Oversample generously; probing does the filtering."""
    ds = _hf_load("akariasai/PopQA", split=split)
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out, used = [], 0
    for i in idx:
        if used >= n:
            break
        ex = ds[i]
        q = ex.get("question", "").strip()
        # PopQA 'possible_answers' is a JSON list string; obj is the entity
        gold = ex.get("obj", "") or ""
        aliases = [str(gold)] if gold else []
        pa = ex.get("possible_answers")
        if pa:
            try:
                import json as _json
                aliases += [str(x) for x in _json.loads(pa)]
            except Exception:
                pass
        if not q:
            continue
        out.append(Instance(
            id=f"popqa_{i}", item_type="fact", probe_kind="parametric_fact",
            context="", question=q, target=str(gold),
            answer_aliases=aliases or [str(gold)],
            claims=[Claim(text=str(gold), source="pending", conflict=0, labeled=True)],
        ))
        used += 1
    return out


# ---------------------------------------------------------------------------
# ConflictQA -> conflict items (counter_memory = verbatim context)
# ---------------------------------------------------------------------------
def load_conflictqa(n=1500, seed=2, config=None) -> list[Instance]:
    """Each item: question + counter_memory (evidence for the NON-parametric
    answer). We present counter_memory as context and expect the model to follow
    it (verbatim). ground_truth / memory_answer = the parametric temptation."""
    # ConflictQA ships several configs (per base LLM). Use one; they share schema.
    cfg = config or "ConflictQA-popQA-chatgpt"
    ds = None
    for sp in ("train", "test", "validation"):
        try:
            ds = _hf_load("osunlp/ConflictQA", config=cfg, split=sp)
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError(
            "Could not load osunlp/ConflictQA. If the parquet mirror lacks the "
            "config, load without config or generate conflicts synthetically "
            "(see load_conflictqa_synthetic fallback).")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out, used = [], 0
    for i in idx:
        if used >= n:
            break
        ex = ds[i]
        q = (ex.get("question") or "").strip()
        counter_ctx = (ex.get("counter_memory") or "").strip()
        counter_ans = (ex.get("counter_answer") or "").strip()
        gt = ex.get("ground_truth")
        param_lure = (gt[0] if isinstance(gt, list) and gt else str(gt or "")).strip()
        if not (q and counter_ctx and counter_ans):
            continue
        # verbatim answer = what the counter-evidence context asserts
        out.append(Instance(
            id=f"conflictqa_{i}", item_type="conflict", probe_kind="conflict",
            context=counter_ctx,
            question=f"According to the passage, {q[0].lower() + q[1:]}",
            target=counter_ans,
            parametric_lure=param_lure or None,
            claims=[Claim(text=counter_ans, source="verbatim", conflict=1, labeled=True)],
            context_char_span=[0, len(counter_ctx)],
        ))
        used += 1
    return out


def load_conflictqa_synthetic(n=1500, seed=2) -> list[Instance]:
    """Fallback if osunlp/ConflictQA can't be loaded: build conflict items by
    presenting a passage that asserts a DIFFERENT answer than the true one.
    The model should follow the passage (verbatim); its parametric answer is the
    lure. Mirrors the ClashEval/ConflictQA construction without the download."""
    facts = load_triviaqa_facts(n * 2, seed=seed)
    rng = random.Random(seed)
    answers = [f.target for f in facts if f.target]
    out = []
    for k, f in enumerate(facts):
        if len(out) >= n:
            break
        alt = rng.choice(answers)
        if alt.strip().lower() == f.target.strip().lower():
            continue
        ctx = (f"According to the reference document, the answer to the following "
               f"question is {alt}. This is stated explicitly in the source.")
        out.append(Instance(
            id=f"synthconflict_{k}", item_type="conflict", probe_kind="conflict",
            context=ctx,
            question=f"According to the passage, {f.question[0].lower() + f.question[1:]}",
            target=alt,
            parametric_lure=f.target,
            claims=[Claim(text=alt, source="verbatim", conflict=1, labeled=True)],
            context_char_span=[0, len(ctx)],
        ))
    return out


# ---------------------------------------------------------------------------
# Natural Questions -> OOD evaluation pool (NOT for training)
# ---------------------------------------------------------------------------
def load_nq_ood(n=2000, seed=3, split="validation") -> list[Instance]:
    """OOD fact pool for generalization testing. Kept separate from training."""
    ds = _hf_load("google-research-datasets/nq_open", split=split)
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out, used = [], 0
    for i in idx:
        if used >= n:
            break
        ex = ds[i]
        q = (ex.get("question") or "").strip()
        ans_list = ex.get("answer") or []
        ans = ans_list[0] if ans_list else ""
        if not (q and ans):
            continue
        out.append(Instance(
            id=f"nq_{i}", item_type="fact", probe_kind="parametric_fact",
            context="", question=q, target=str(ans),
            claims=[Claim(text=str(ans), source="pending", conflict=0, labeled=True)],
        ))
        used += 1
    return out


# ---------------------------------------------------------------------------
# Assemble the full external pool
# ---------------------------------------------------------------------------
def build_external(n_trivia=3000, n_popqa=3000, n_conflict=1500, n_nq=2000,
                   seed=0) -> dict[str, list[Instance]]:
    """Return dict of pools. Oversampled; probing filters facts down to clean
    parametric/confabulated with a gray-zone dropped."""
    return {
        "trivia": load_triviaqa_facts(n_trivia, seed=seed),
        "popqa": load_popqa_facts(n_popqa, seed=seed + 1),
        "conflict": load_conflictqa(n_conflict, seed=seed + 2),
        "nq_ood": load_nq_ood(n_nq, seed=seed + 3),
    }


if __name__ == "__main__":
    # smoke test: just report sizes (requires `datasets` + internet)
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_trivia", type=int, default=50)
    p.add_argument("--n_popqa", type=int, default=50)
    p.add_argument("--n_conflict", type=int, default=50)
    p.add_argument("--n_nq", type=int, default=50)
    args = p.parse_args()
    pools = build_external(args.n_trivia, args.n_popqa, args.n_conflict, args.n_nq)
    for k, v in pools.items():
        print(f"{k:10s}: {len(v)} instances")
        if v:
            ex = v[0]
            print(f"   e.g. [{ex.probe_kind}] Q: {ex.question[:70]}")
