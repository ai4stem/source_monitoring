"""
signals.py
==========
Token-level source signals and claim-level pooling, implementing the maths we
fixed in the framework:

  context mass        m_i    = sum_h w_h * sum_{k in C} a^{(h)}_{i->k}
  normalized entropy  H~_i   = H_i / log(n_i),   H_i = -sum_k abar_{i->k} log abar
  cross-head consensus c_i   = mean pairwise cosine similarity of heads' attn rows

  claim pooling       u_j    = sum_{i in T_j} alpha_ji * z_i   (attention-weighted)

The functions operate on plain numpy attention arrays so they can be unit-tested
without a GPU.  `extract_attention_for_instance` (which actually runs a HF model)
is import-guarded and only needed on the training server.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Core signal maths  (numpy -- testable anywhere)
# ---------------------------------------------------------------------------

def context_mass(attn: np.ndarray, context_idx: np.ndarray,
                 head_weights: np.ndarray | None = None) -> np.ndarray:
    """m_i for every query token.

    attn: [n_heads, n_query, n_key] attention weights (softmax'd, rows sum~1).
    context_idx: indices (into key axis) that belong to the evidence span C.
    head_weights: [n_heads] softmaxed weights w_h; uniform if None.
    returns: [n_query] context mass in [0,1].
    """
    n_heads = attn.shape[0]
    if head_weights is None:
        head_weights = np.ones(n_heads) / n_heads
    # mass to context per head: [n_heads, n_query]
    mass_h = attn[:, :, context_idx].sum(axis=2)
    # weight heads
    return (head_weights[:, None] * mass_h).sum(axis=0)


def normalized_entropy(attn: np.ndarray, valid_counts: np.ndarray | None = None,
                       eps: float = 1e-12) -> np.ndarray:
    """H~_i for every query token, averaged over heads then length-normalized.

    attn: [n_heads, n_query, n_key].
    valid_counts: [n_query] number of attendable keys n_i (causal mask aware).
                  If None, uses n_key for all (non-causal).
    returns: [n_query] in [0,1].
    """
    abar = attn.mean(axis=0)                      # [n_query, n_key]
    abar = abar / (abar.sum(axis=1, keepdims=True) + eps)
    H = -(abar * np.log(abar + eps)).sum(axis=1)  # [n_query]
    n_query, n_key = abar.shape
    if valid_counts is None:
        valid_counts = np.full(n_query, n_key)
    denom = np.log(np.maximum(valid_counts, 2))   # avoid log(1)=0
    return H / denom


def cross_head_consensus(attn: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """c_i = mean pairwise cosine similarity across heads' attention rows.

    High consensus -> heads agree -> reliable. Low -> disagreement -> confab-ish.
    returns: [n_query] in [0,1].
    """
    n_heads, n_query, n_key = attn.shape
    if n_heads == 1:
        return np.ones(n_query)
    # normalize each head's row to unit length
    norm = attn / (np.linalg.norm(attn, axis=2, keepdims=True) + eps)
    out = np.zeros(n_query)
    pairs = 0
    for a in range(n_heads):
        for b in range(a + 1, n_heads):
            out += (norm[a] * norm[b]).sum(axis=1)   # cosine per query token
            pairs += 1
    return out / max(pairs, 1)


def token_signals(attn: np.ndarray, context_idx: np.ndarray,
                  valid_counts: np.ndarray | None = None,
                  head_weights: np.ndarray | None = None,
                  use_consensus: bool = True) -> np.ndarray:
    """Stack per-token signals z_i = [m_i, H~_i, (c_i)].

    returns: [n_query, d] with d=3 (or 2 without consensus).
    """
    m = context_mass(attn, context_idx, head_weights)
    h = normalized_entropy(attn, valid_counts)
    cols = [m, h]
    if use_consensus:
        cols.append(cross_head_consensus(attn))
    return np.stack(cols, axis=1)


# ---------------------------------------------------------------------------
# Claim-level pooling  (attention-weighted; learnable g(.) lives in the model,
# but we provide a parameter-free softmax-pool for analysis/inspection).
# ---------------------------------------------------------------------------

def attention_pool(z: np.ndarray, claim_token_idx: np.ndarray,
                   score: np.ndarray | None = None) -> np.ndarray:
    """u_j for a single claim span.

    z: [n_query, d] token signals.
    claim_token_idx: indices of tokens in T_j.
    score: optional [n_query] pooling logits g(z_i); if None, score = -H~ so that
           focused (low-entropy) tokens get more weight (a sensible default proxy).
    returns: [d] pooled claim signal.
    """
    zc = z[claim_token_idx]                        # [|T_j|, d]
    if score is None:
        # default: weight by focus (1 - normalized entropy, col index 1)
        s = 1.0 - zc[:, 1]
    else:
        s = score[claim_token_idx]
    s = s - s.max()
    w = np.exp(s)
    w = w / (w.sum() + 1e-12)
    return (w[:, None] * zc).sum(axis=0)


# ---------------------------------------------------------------------------
# HF model attention extraction  (server-only; import-guarded)
# ---------------------------------------------------------------------------

def extract_attention_for_instance(model, tokenizer, instance, layer: int = -1,
                                   device: str = "cuda", max_new_tokens: int = 24):
    """Generate the answer, then measure attention over the ANSWER tokens.

    The source of a *claim* depends on where the model looks WHILE PRODUCING the
    answer -- not on a placeholder token of the prompt. So we:
      1. build the prompt (context + question, chat-templated),
      2. generate a short answer,
      3. re-run the full (prompt+answer) sequence with output_attentions,
      4. return attention plus the KEY indices of the context span and the
         QUERY indices of the generated answer tokens (the claim span).

    Returns dict with:
        attn          [n_heads, q, k]
        context_idx   key indices belonging to the evidence/context span
        claim_idx     query indices of the generated answer tokens
        valid_counts  causal attendable counts per query
    Server-only (needs torch).
    """
    import torch  # noqa

    # ---- 1. build prompt (chat template + thinking off, mirroring probing) ----
    user_text = (instance.context + "\n" + instance.question).strip() \
        if instance.context else instance.question
    prompt = user_text
    if getattr(tokenizer, "chat_template", None):
        for kw in ({"enable_thinking": False}, {}):
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_text}],
                    tokenize=False, add_generation_prompt=True, **kw)
                break
            except TypeError:
                continue

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]

    # ---- 2. generate a short answer ----
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    full_ids = gen[0]                       # [prompt_len + answer_len]
    answer_len = full_ids.shape[0] - prompt_len
    if answer_len <= 0:
        # model emitted nothing; fall back to last prompt token as claim
        answer_len = 1

    # ---- 3. re-run full sequence to get attentions over prompt+answer ----
    with torch.no_grad():
        out = model(input_ids=full_ids.unsqueeze(0), output_attentions=True)
    attn = out.attentions[layer][0].float().cpu().numpy()   # [n_heads, q, k]
    n = full_ids.shape[0]

    # ---- 4. locate context span (keys) and answer span (queries) ----
    # context span: the context portion of the prompt, if any
    if instance.context:
        ctx_ids = tokenizer(instance.context, add_special_tokens=False)["input_ids"]
        ctx_len = min(len(ctx_ids), prompt_len)
    else:
        ctx_len = 0
    context_idx = np.arange(ctx_len) if ctx_len > 0 else np.arange(0)

    # claim span: the generated ANSWER tokens (queries)
    claim_start = prompt_len
    claim_idx = np.arange(claim_start, n)
    if claim_idx.size == 0:
        claim_idx = np.array([n - 1])

    valid_counts = np.arange(1, n + 1)
    return {
        "attn": attn,
        "token_ids": full_ids.cpu().tolist(),
        "context_idx": context_idx,
        "claim_idx": claim_idx,
        "valid_counts": valid_counts,
    }
