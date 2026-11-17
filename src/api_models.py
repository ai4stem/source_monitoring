"""
api_models.py
=============
Commercial LLM comparison via PROMPT DESIGN ONLY (no weight access).

Since GPT-4o / Claude expose no attention, we cannot compute m_i / H~_i. Instead
we elicit source attribution behaviorally:
  - For DRM probes: did the model claim the word was present? (-> confab rate)
  - For conflict probes: did it follow the context (verbatim) or its prior?
  - For source self-report: we ask it to label its answer's source as one of
    {from the passage / from my own knowledge / I'm not sure / guessing}, which
    we map onto {verbatim / parametric / confabulated}.

This is the same prompt-extraction approach used for SOTA models in the prior
work, and carries the SAME caveat: verbalized self-reports are biased and are
not directly comparable to attention-derived signals. Report them as a separate
behavioral comparison, not as identical measurements.

Responses are cached to disk keyed by (model_id, prompt_hash) so repeated runs
on the frozen dataset cost nothing and stay reproducible.
"""

from __future__ import annotations
import hashlib
import json
import os
import re
import time
import env_load  # noqa: F401  -- auto-loads .env on import (sets API keys)

CACHE_DIR = "outputs/api_cache"

# Map self-reported source phrases -> our 3 classes
SOURCE_PROMPT = """Answer the question, then on a new line declare the SOURCE of your answer using EXACTLY one of these tags:
[SOURCE: passage]   - the answer came from information stated in the passage above
[SOURCE: knowledge] - the answer came from your own prior knowledge
[SOURCE: guess]     - you are not sure / you are guessing

Question: {question}"""

SOURCE_MAP = {
    "passage": "verbatim",
    "knowledge": "parametric",
    "guess": "confabulated",
}


def _prompt_hash(model_id: str, prompt: str) -> str:
    return hashlib.sha256((model_id + "||" + prompt).encode()).hexdigest()[:24]


def _cache_path(model_id: str, prompt: str) -> str:
    return os.path.join(CACHE_DIR, f"{model_id}_{_prompt_hash(model_id, prompt)}.json")


def _cached(model_id: str, prompt: str):
    p = _cache_path(model_id, prompt)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)["response"]
    return None


def _store(model_id: str, prompt: str, response: str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_cache_path(model_id, prompt), "w", encoding="utf-8") as f:
        json.dump({"model_id": model_id, "prompt": prompt, "response": response}, f)


# ---------------------------------------------------------------------------
# Provider calls (import-guarded; needs `openai` / `anthropic` + API keys)
# ---------------------------------------------------------------------------

def call_openai(model_id: str, prompt: str, temperature: float = 0.0,
                max_tokens: int = 200, retries: int = 3) -> str:
    import openai
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    msgs = [{"role": "user", "content": prompt}]
    for attempt in range(retries):
        try:
            # newer models (GPT-5+) use max_completion_tokens and reject custom
            # temperature; try that first, fall back to legacy params.
            try:
                r = client.chat.completions.create(
                    model=model_id, messages=msgs,
                    max_completion_tokens=max_tokens, timeout=30)
            except openai.BadRequestError:
                r = client.chat.completions.create(
                    model=model_id, messages=msgs,
                    max_tokens=max_tokens, temperature=temperature, timeout=30)
            return r.choices[0].message.content
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def call_anthropic(model_id: str, prompt: str, temperature: float = 0.0,
                   max_tokens: int = 200, retries: int = 3) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    msgs = [{"role": "user", "content": prompt}]
    for attempt in range(retries):
        try:
            # some newer models deprecate `temperature`; try without it first,
            # fall back to including it for older models.
            try:
                r = client.messages.create(
                    model=model_id, max_tokens=max_tokens, messages=msgs)
            except anthropic.BadRequestError:
                r = client.messages.create(
                    model=model_id, max_tokens=max_tokens,
                    temperature=temperature, messages=msgs)
            return r.content[0].text
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def query(spec, prompt: str, use_cache: bool = True) -> str:
    """Dispatch to the right provider, with caching keyed on the frozen prompt."""
    mid = spec.hf_name_or_api_id
    if use_cache:
        c = _cached(mid, prompt)
        if c is not None:
            return c
    if spec.api_provider == "openai":
        resp = call_openai(mid, prompt)
    elif spec.api_provider == "anthropic":
        resp = call_anthropic(mid, prompt)
    else:
        raise ValueError(f"unknown api provider: {spec.api_provider}")
    _store(mid, prompt, resp)
    return resp


# ---------------------------------------------------------------------------
# Response parsing -> behavioral fields + self-reported source
# ---------------------------------------------------------------------------

def parse_source_tag(response: str) -> str | None:
    m = re.search(r"\[source:\s*(passage|knowledge|guess)\]", response.lower())
    if m:
        return SOURCE_MAP.get(m.group(1))
    return None


def parse_yes_no(response: str) -> bool:
    """True if the model affirms presence (for DRM probes)."""
    head = response.strip().lower()[:40]
    # check negation FIRST -- "not present" must not match "present"
    if re.search(r"\b(no|not|wasn'?t|isn'?t|absent|never)\b", head):
        return False
    if re.search(r"\b(yes|correct|present|it was|appeared)\b", head):
        return True
    return "yes" in head  # fallback


def evaluate_api_instance(spec, instance) -> dict:
    """Run one frozen instance through an API model; return a prediction record
    matching analyze.py's schema (behavioral fields + self-reported source)."""
    claim = instance.claims[0] if instance.claims else None
    pk = instance.probe_kind

    # Build the prompt with context + question + source self-report request.
    base = (instance.context + "\n\n" if instance.context else "")
    prompt = base + SOURCE_PROMPT.format(question=instance.question)
    resp = query(spec, prompt)

    rec = {
        "instance_id": instance.id, "probe_kind": pk, "item_type": instance.item_type,
        "model": spec.key, "access": "api",
        "true_source": claim.source if (claim and claim.is_resolved()) else None,
        "pred_source": parse_source_tag(resp),     # self-reported
        "list_type": instance.list_type,
        "raw_response": resp[:300],
    }
    # behavioral fields for DRM / conflict metrics
    if pk in ("critical_lure", "studied", "unrelated"):
        rec["answered_yes"] = parse_yes_no(resp)
    if pk == "conflict":
        rec["followed_context"] = (instance.target.lower() in resp.lower())
    elif pk == "non_conflict_control":
        rec["followed_context"] = (instance.target.lower() in resp.lower())
    rec["task_correct"] = (instance.target.lower() in resp.lower())
    return rec
