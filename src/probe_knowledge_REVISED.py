"""
probe_knowledge.py  (REVISED — saves raw samples for baselines & grey-zone analysis)
===================================================================================
Model-relative parametric labeling via behavioral stability.

CHANGES vs original (all backward compatible):
  * stability_record now optionally stores the RAW per-sample data:
      - answers      : list[str]  (all K*R generated answers)
      - correct      : list[bool] (per-sample correctness)
      - by_paraphrase: list[dict] (per-paraphrase answers+correct, for self-consistency)
    This is needed for (2) external baselines: self-consistency and semantic entropy.
  * probe_fact returns the enriched record; the classification is unchanged, so
    parametric/gray/confabulated labels are IDENTICAL to before.
  * grey-zone ('prompt_sensitive') items keep their stability record too, so the
    grey-zone sensitivity analysis (1) can read accuracy/answers for them.

The classification thresholds and logic are UNCHANGED, so re-running reproduces
the same labels while additionally persisting the raw signals we previously threw
away.
"""
from __future__ import annotations
from collections import Counter
import math


def answer_entropy(answers: list[str]) -> float:
    """Shannon entropy over normalized answer strings (nats)."""
    if not answers:
        return 0.0
    norm = [a.strip().lower() for a in answers]
    counts = Counter(norm)
    total = len(norm)
    return -sum((c / total) * math.log(c / total) for c in counts.values())


def classify_stability(accuracy: float, ans_entropy: float = None,
                       acc_hi: float = 0.6, acc_lo: float = 0.2) -> str:
    """Classify a fact by CORRECTNESS CONSISTENCY across paraphrases/samples.

    'prompt_agnostic'  (parametric):   reliably correct   -> accuracy >= acc_hi
    'unanswerable'     (confabulated):  reliably wrong     -> accuracy <= acc_lo
    'prompt_sensitive' (gray zone):     inconsistent       -> in between
    """
    if accuracy >= acc_hi:
        return "prompt_agnostic"
    if accuracy <= acc_lo:
        return "unanswerable"
    return "prompt_sensitive"


def stability_record(answers: list[str], correct: list[bool],
                     by_paraphrase: list[dict] | None = None,
                     keep_raw: bool = True) -> dict:
    """Build the stability record.

    keep_raw=True additionally stores the per-sample answers and correctness and,
    if provided, the per-paraphrase breakdown. Set keep_raw=False to reproduce the
    old compact record exactly.
    """
    acc = sum(correct) / max(len(correct), 1)
    ent = answer_entropy(answers)
    cat = classify_stability(acc, ent)
    rec = {"n_samples": len(answers), "accuracy": round(acc, 3),
           "answer_entropy": round(ent, 3), "category": cat}
    if keep_raw:
        rec["answers"] = answers
        rec["correct"] = [bool(c) for c in correct]
        if by_paraphrase is not None:
            rec["by_paraphrase"] = by_paraphrase
    return rec


def apply_stability_label(instance, record: dict) -> None:
    """Mutate a fact Instance's claim source based on a stability record."""
    cat = record["category"]
    for c in instance.claims:
        if cat == "prompt_agnostic":
            c.source, c.labeled = "parametric", True
        elif cat == "unanswerable":
            c.source, c.labeled = "confabulated", True
        else:  # prompt_sensitive -> gray zone, excluded from loss
            c.source, c.labeled = "pending", False
    instance.stability = record


# ---------------------------------------------------------------------------
# Generation (server-only; import-guarded)
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    import re
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _is_correct(gold, text: str) -> bool:
    golds = gold if isinstance(gold, (list, tuple)) else [gold]
    t = _normalize(text)
    for g in golds:
        g = _normalize(str(g))
        if g and g in t:
            return True
    return False


def probe_fact(model, tokenizer, question: str, gold,
               paraphraser=None, n_paraphrase: int = 8, n_sample: int = 5,
               device: str = "cuda", keep_raw: bool = True) -> dict:
    """Closed-book stability probe for one fact. Requires torch.

    REVISED: records per-paraphrase answers so self-consistency and semantic
    entropy can be computed later. Labels are unchanged.
    """
    import torch  # noqa

    def default_paraphrases(q: str) -> list[str]:
        ql = q[0].lower() + q[1:] if q else q
        variants = [
            q,
            f"{q} Answer in a few words.",
            f"Please answer concisely: {q}",
            f"Can you tell me {ql}",
            f"{q} Give only the answer.",
            f"Answer briefly: {q}",
        ]
        return variants[:n_paraphrase]

    def format_prompt(text: str) -> str:
        if hasattr(tokenizer, "apply_chat_template") and \
           getattr(tokenizer, "chat_template", None):
            for kwargs in ({"enable_thinking": False}, {}):
                try:
                    return tokenizer.apply_chat_template(
                        [{"role": "user", "content": text}],
                        tokenize=False, add_generation_prompt=True, **kwargs)
                except TypeError:
                    continue
                except Exception:
                    return text
        return text

    def strip_think(t: str) -> str:
        import re
        t = re.sub(r"<think>.*?</think>", " ", t, flags=re.DOTALL)
        t = re.sub(r"<think>.*", " ", t, flags=re.DOTALL)
        return t.strip()

    raw_prompts = (paraphraser or default_paraphrases)(question)
    prompts = [format_prompt(p) for p in raw_prompts]

    answers, correct = [], []
    by_paraphrase = []
    for rawp, p in zip(raw_prompts, prompts):
        pa_answers, pa_correct = [], []
        for _ in range(max(1, n_sample // len(prompts) + 1)):
            enc = tokenizer(p, return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=96, do_sample=True,
                                     temperature=0.7, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(gen[0][enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip()
            text = strip_think(text)
            ok = _is_correct(gold, text)
            answers.append(text); correct.append(ok)
            pa_answers.append(text); pa_correct.append(ok)
        by_paraphrase.append({"paraphrase": rawp,
                              "answers": pa_answers,
                              "correct": [bool(c) for c in pa_correct]})

    return stability_record(answers, correct,
                            by_paraphrase=by_paraphrase, keep_raw=keep_raw)
