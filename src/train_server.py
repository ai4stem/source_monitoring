"""
train_server.py  (SERVER-SIDE -- now with a real training loop)
==============================================================
LoRA/QLoRA fine-tuning of the 3-way source-monitoring head with our composite
loss  L = L_LM + alpha*L_source + gamma*L_conflict.

Design (mirrors your prior Predictive-Metacognition trainer, retargeted):
  - wrap a LoRA base model in `SourceMonitoringModel`
  - in forward: run base with output_attentions/hidden_states, compute token
    signals (context mass m, normalized entropy H~) for the answer span, pool to
    a claim vector u, push through the SourceHead -> 3 logits
  - composite loss via model.composite_loss_torch
  - a custom Trainer.compute_loss feeds claim labels + conflict mask

torch/transformers/peft are imported lazily; this file is import-safe without
them (so the rest of the package and tests don't need a GPU), but train()
requires the server stack.

Run:
  python train_server.py --model qwen3-8b
  python train_server.py --model qwen3-32b      # QLoRA, device_map shards 4 GPUs
"""

from __future__ import annotations
import os
import env_load  # noqa: F401  -- auto-loads .env (HF_TOKEN for gated models)

import numpy as np
from schema import load_dataset, save_dataset
from frozen_dataset import load_frozen, FROZEN_PATH
from build_dataset import build_all
from model import (build_source_head, composite_loss_torch, class_weights,
                   CLASS_TO_IDX, SOURCE_CLASSES)
from probe_knowledge import probe_fact, apply_stability_label
from checkpoint import save_checkpoint
from distributed import load_distributed, attach_lora


# ---------------------------------------------------------------------------
# Stage 2: model-relative parametric labeling (full/half precision, NOT 4-bit)
# ---------------------------------------------------------------------------

def assign_parametric_labels(model, tokenizer, instances, device="cuda"):
    for inst in instances:
        if inst.item_type == "fact" and inst.probe_kind == "parametric_fact":
            rec = probe_fact(model, tokenizer, inst.question, inst.target, device=device)
            apply_stability_label(inst, rec)
    return instances


# ---------------------------------------------------------------------------
# Tokenization: build a prompt+answer, mark the answer span as the claim span
# and the context span as the evidence region C (for context-mass signal).
# ---------------------------------------------------------------------------

def encode_instance(tokenizer, inst, max_len=256):
    """Return input_ids, attention_mask, lm_labels, plus index bookkeeping:
       context_idx (keys in C), answer_start (first answer token position)."""
    ctx = (inst.context + "\n") if inst.context else ""
    prompt = ctx + "Q: " + inst.question + "\nA: "
    answer = inst.target

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_len]

    # LM labels: -100 on the prompt, real ids on the answer (standard SFT mask)
    lm_labels = [-100] * len(prompt_ids) + answer_ids
    lm_labels = lm_labels[:max_len]

    # context key indices = tokens belonging to inst.context at the start
    ctx_len = len(tokenizer(ctx, add_special_tokens=False)["input_ids"]) if ctx else 0
    context_idx = list(range(min(ctx_len, len(input_ids))))

    answer_start = min(len(prompt_ids), len(input_ids) - 1)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "lm_labels": lm_labels,
        "context_idx": context_idx,
        "answer_start": answer_start,
    }


def build_training_examples(tokenizer, instances, max_len=256):
    """Keep only claims with a resolved source label; attach class index + conflict."""
    examples = []
    for inst in instances:
        claim = inst.claims[0] if inst.claims else None
        if claim is None or not claim.is_resolved():
            continue
        enc = encode_instance(tokenizer, inst, max_len)
        enc["source_label"] = CLASS_TO_IDX[claim.source]
        enc["conflict"] = int(claim.conflict)
        examples.append(enc)
    return examples


# ---------------------------------------------------------------------------
# Collator: pad and stack, carry per-example index metadata as python lists
# ---------------------------------------------------------------------------

def make_collator(tokenizer):
    import torch

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = tokenizer.pad_token_id or 0

        def padlist(key, fill):
            return torch.tensor(
                [b[key] + [fill] * (maxlen - len(b[key])) for b in batch])

        out = {
            "input_ids": padlist("input_ids", pad),
            "attention_mask": padlist("attention_mask", 0),
            "lm_labels": padlist("lm_labels", -100),
            "source_label": torch.tensor([b["source_label"] for b in batch]),
            "conflict": torch.tensor([b["conflict"] for b in batch]),
            # index metadata kept as python lists (ragged)
            "context_idx": [b["context_idx"] for b in batch],
            "answer_start": [b["answer_start"] for b in batch],
        }
        return out
    return collate


# ---------------------------------------------------------------------------
# Wrapped model: base (LoRA) + SourceHead, computing signals in forward
# ---------------------------------------------------------------------------

def build_wrapped_model(peft_model, signal_dim=3, use_hidden_state=False):
    import torch
    import torch.nn as nn
    from signals import context_mass, normalized_entropy, cross_head_consensus

    class SourceMonitoringModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = peft_model
            self.head = build_source_head(signal_dim=signal_dim,
                                          use_hidden_state=use_hidden_state)
            # learnable per-head weights phi_h (filled lazily once n_heads known)
            self.phi = None

        def _token_signals_batch(self, attentions, context_idx_list, answer_start_list):
            """Compute pooled claim signal u for each example in the batch.
            attentions: last-layer attn [B, n_heads, q, k] (torch tensor)."""
            attn = attentions[-1] if isinstance(attentions, (list, tuple)) else attentions
            B, H, Q, K = attn.shape
            if self.phi is None:
                self.phi = nn.Parameter(torch.zeros(H, device=attn.device))
            head_w = torch.softmax(self.phi, dim=0).detach().cpu().numpy()

            us = []
            attn_np = attn.detach().float().cpu().numpy()
            for b in range(B):
                cidx = np.array(context_idx_list[b], dtype=int)
                a_start = answer_start_list[b]
                # claim tokens = answer span (a_start .. end of valid tokens)
                valid = np.arange(1, Q + 1)
                a = attn_np[b]                      # [H, Q, K]
                if len(cidx) == 0:
                    cidx = np.array([0])
                m = context_mass(a, cidx, head_w)   # [Q]
                h = normalized_entropy(a, valid)    # [Q]
                c = cross_head_consensus(a)         # [Q]
                z = np.stack([m, h, c], axis=1)     # [Q, 3]
                claim_tok = np.arange(a_start, Q)
                if len(claim_tok) == 0:
                    claim_tok = np.array([Q - 1])
                # focus-weighted pooling (1 - entropy)
                s = 1.0 - z[claim_tok, 1]
                s = s - s.max()
                w = np.exp(s); w = w / (w.sum() + 1e-12)
                u = (w[:, None] * z[claim_tok]).sum(0)
                us.append(u)
            return torch.tensor(np.stack(us), dtype=torch.float32, device=attn.device)

        def forward(self, input_ids, attention_mask, lm_labels=None,
                    source_label=None, conflict=None,
                    context_idx=None, answer_start=None,
                    alpha=0.1, gamma=0.02, beta=None, **kw):
            out = self.base(input_ids=input_ids, attention_mask=attention_mask,
                            output_attentions=True, output_hidden_states=False)
            # LM loss (standard SFT shift)
            from torch.nn import CrossEntropyLoss
            logits = out.logits
            lm_loss = torch.zeros((), device=logits.device)
            if lm_labels is not None:
                sl = logits[..., :-1, :].contiguous()
                tl = lm_labels[..., 1:].contiguous()
                lm_loss = CrossEntropyLoss(ignore_index=-100)(
                    sl.view(-1, sl.size(-1)), tl.view(-1))

            # source signals -> head logits. Move u to the head's device (the
            # input/embedding GPU), so src_logits co-locates with source_label.
            u = self._token_signals_batch(out.attentions, context_idx, answer_start)
            head_device = next(self.head.parameters()).device
            u = u.to(head_device)
            src_logits = self.head(u)               # [B, 3]

            total, parts = lm_loss, {"lm": float(lm_loss.detach())}
            if source_label is not None and beta is not None:
                source_label = source_label.to(head_device)
                if conflict is not None:
                    conflict = conflict.to(head_device)
                cmask = (conflict == 1) if conflict is not None else \
                    torch.zeros_like(source_label, dtype=torch.bool)
                lm_for_sum = lm_loss.to(head_device)
                total, parts = composite_loss_torch(
                    lm_for_sum, src_logits, source_label, cmask, beta.to(head_device),
                    alpha=alpha, gamma=gamma)
            return {"loss": total, "parts": parts, "source_logits": src_logits}

        def save_pretrained(self, d, **kw):
            self.base.save_pretrained(d, **kw)

        def gradient_checkpointing_enable(self, **kw):
            if hasattr(self.base, "gradient_checkpointing_enable"):
                self.base.gradient_checkpointing_enable()

    return SourceMonitoringModel()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model_name, quantize=False, alpha=0.1, gamma=0.02, epochs=3,
          lr=3e-4, ckpt_dir="outputs/ckpt", max_len=256, batch_size=2,
          grad_accum=8, dataset_path=FROZEN_PATH, device="cuda",
          large_scale=False, n_squad=0, n_trivia=4000, n_popqa=2000, n_conflict=1500,
          target_parametric=2500, target_confab=800, target_conflict=1000,
          n_paraphrase=6, probe_only=False):
    import torch
    from torch.utils.data import DataLoader
    from registry import get_model

    # spec lookup by hf name (train() may be called directly)
    spec = next((m for m in __import__("registry").MODELS
                 if m.hf_name_or_api_id == model_name), None)

    # --- load base (full precision for probing) ---
    print(f"[train] loading {model_name} (probe pass, full precision)")
    probe_kwargs = type("S", (), {"hf_name_or_api_id": model_name,
                                  "quantize_4bit": False, "access": "weights",
                                  "distributed": (spec.distributed if spec else False)})
    base_probe, tok = load_distributed(probe_kwargs, for_training=False)

    # --- assemble training data ---
    # If a frozen LARGE pool exists, load its fixed question set and probe labels
    # on THIS model (model-relative). Otherwise fall back to the small built-in set.
    if large_scale:
        from assemble_dataset import assemble_full
        import os
        mkey = spec.key if spec else "model"
        probe_cache = f"outputs/probed_{mkey}.jsonl"
        eval_path = f"outputs/eval_{mkey}.jsonl"
        instances, eval_set, ood_pool, report = assemble_full(
            base_probe, tok, device=device,
            n_squad=n_squad, n_trivia=n_trivia, n_popqa=n_popqa, n_conflict=n_conflict,
            target_parametric=target_parametric, target_confab=target_confab,
            target_conflict=target_conflict,
            n_paraphrase=n_paraphrase, seed=0,
            probe_cache=probe_cache, eval_frac=0.1)
        # save held-out eval set (used by run_comparison instead of the tiny
        # frozen set) and the shared OOD pool
        os.makedirs("outputs", exist_ok=True)
        save_dataset(eval_set, eval_path)
        print(f"[train] saved eval set -> {eval_path}")
        os.makedirs("data/frozen", exist_ok=True)
        if not os.path.exists("data/frozen/ood_pool.jsonl"):
            save_dataset(ood_pool, "data/frozen/ood_pool.jsonl")
    else:
        try:
            instances = load_frozen(dataset_path)
        except Exception:
            instances = build_all()
        instances = assign_parametric_labels(base_probe, tok, instances, device=device)

    del base_probe
    torch.cuda.empty_cache()

    # probe-only mode: eval set is already saved above; skip training. Used for
    # hidden-state / signal analysis on the size ladder, where the LoRA source
    # head isn't needed.
    if probe_only:
        print(f"[train] probe_only=True -> probing done, eval set saved, "
              f"skipping training for {model_name}")
        return

    # --- class weights from resolved labels ---
    from collections import Counter
    counts = Counter(c.source for i in instances for c in i.claims if c.is_resolved())
    beta_np = class_weights(dict(counts))
    beta = torch.tensor(beta_np, dtype=torch.float32, device=device)
    print(f"[train] class weights {dict(zip(SOURCE_CLASSES, beta_np.round(3)))}")

    # --- reload base for training (4-bit if requested) ---
    train_spec = spec if spec else type("S", (), {
        "hf_name_or_api_id": model_name, "quantize_4bit": quantize,
        "access": "weights", "distributed": False})
    base, tok = load_distributed(train_spec, for_training=True)
    peft = attach_lora(base, prepare_kbit=quantize)
    wrapped = build_wrapped_model(peft)

    # Determine the device where model INPUTS must live (the input-embedding
    # device). For a single-GPU model this is cuda:0; for a sharded model it is
    # wherever the embedding layer landed. Batches and the source head go here.
    try:
        in_device = next(wrapped.base.parameters()).device
        # more precise: the embedding device if we can find it
        emb = wrapped.base.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            in_device = emb.weight.device
    except Exception:
        in_device = torch.device(device)
    print(f"[train] input device = {in_device}")
    wrapped.head = wrapped.head.to(in_device)
    beta = beta.to(in_device)

    # --- examples + loader ---
    examples = build_training_examples(tok, instances, max_len)
    print(f"[train] {len(examples)} labeled training claims")
    loader = DataLoader(examples, batch_size=batch_size, shuffle=True,
                        collate_fn=make_collator(tok))

    # --- optimizer over LoRA + head + phi ---
    params = [p for p in wrapped.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    total_steps = max(1, (len(loader) // grad_accum) * epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    wrapped.train()
    step = 0
    for ep in range(epochs):
        for i, batch in enumerate(loader):
            batch = {k: (v.to(in_device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            out = wrapped(**batch, alpha=alpha, gamma=gamma, beta=beta)
            loss = out["loss"] / grad_accum
            loss.backward()
            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % 10 == 0:
                    print(f"[train] ep{ep} step{step} loss={out['loss'].item():.4f} "
                          f"{ {k: round(v,3) for k,v in out['parts'].items()} }")

    # --- save LoRA adapter + source head + config ---
    # Move the small head/phi to CPU before saving -- avoids an OOM on the last
    # step when the GPU is nearly full, and checkpoints are device-agnostic.
    config = {"model_name": model_name, "alpha": alpha, "gamma": gamma, "lr": lr,
              "epochs": epochs, "class_weights": beta_np.tolist(),
              "quantized": bool(quantize)}
    wrapped.head = wrapped.head.to("cpu")
    phi_cpu = wrapped.phi.detach().to("cpu") if wrapped.phi is not None else None
    save_checkpoint(ckpt_dir, peft_model=peft, source_head=wrapped.head,
                    head_weights=phi_cpu, config=config)
    print(f"[train] saved -> {ckpt_dir}")

    # free everything so the next model in run_all.sh starts with a clean GPU
    del wrapped, peft, base, opt
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="registry model key")
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.02)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    # large-scale data (oversample external QA + probe + balance)
    p.add_argument("--large_scale", action="store_true",
                   help="download TriviaQA/PopQA/ConflictQA, probe, and balance")
    p.add_argument("--n_squad", type=int, default=0)
    p.add_argument("--n_trivia", type=int, default=4000)
    p.add_argument("--n_popqa", type=int, default=2000)
    p.add_argument("--n_conflict", type=int, default=1500)
    p.add_argument("--target_parametric", type=int, default=2500)
    p.add_argument("--target_confab", type=int, default=800)
    p.add_argument("--target_conflict", type=int, default=1000)
    p.add_argument("--n_paraphrase", type=int, default=6)
    p.add_argument("--probe_only", action="store_true",
                   help="probe and save eval set, then skip training "
                        "(for hidden-state/signal analysis on the size ladder)")
    args = p.parse_args()

    from registry import get_model
    spec = get_model(args.model)
    ckpt = args.ckpt_dir or f"outputs/ckpt_{spec.key}"
    print(f"[train] model={spec.key} ({spec.hf_name_or_api_id}) -> {ckpt}")
    train(model_name=spec.hf_name_or_api_id, quantize=spec.quantize_4bit,
          alpha=args.alpha, gamma=args.gamma, epochs=args.epochs, lr=args.lr,
          ckpt_dir=ckpt, batch_size=args.batch_size, grad_accum=args.grad_accum,
          large_scale=args.large_scale, n_squad=args.n_squad, n_trivia=args.n_trivia,
          n_popqa=args.n_popqa, n_conflict=args.n_conflict,
          target_parametric=args.target_parametric, target_confab=args.target_confab,
          target_conflict=args.target_conflict, n_paraphrase=args.n_paraphrase,
          probe_only=args.probe_only)
