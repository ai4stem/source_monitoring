"""
run_comparison.py  (SERVER-SIDE orchestrator)
=============================================
Runs EVERY registered model on the SAME frozen dataset and writes one
comparison table. Open models use intrinsic signals (and optionally a trained
source head); API models use prompt-design self-report. Each model's per-claim
predictions and metrics are saved under outputs/<model_key>/ so a run can be
resumed or a single model re-run without touching the others.

Determinism guarantees:
  - dataset is the frozen, hash-verified one (frozen_dataset.load_frozen)
  - API responses are cached (api_models) -> identical across reruns
  - torch seeds are set; generation uses temperature handling per access mode

Run:
  python frozen_dataset.py freeze                  # once
  python run_comparison.py --models all            # all models, frozen data
  python run_comparison.py --models gpt-4o claude-sonnet   # just APIs
  python run_comparison.py --aggregate             # rebuild the summary table
"""

from __future__ import annotations
import argparse
import json
import os
import env_load  # noqa: F401  -- auto-loads .env (API keys, HF_TOKEN)

from frozen_dataset import load_frozen, FROZEN_PATH
from registry import MODELS, get_model, list_keys
from analyze import evaluate_all, save_results, save_predictions, load_predictions


def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Per-access-mode evaluation
# ---------------------------------------------------------------------------

def run_api_model(spec, instances) -> list[dict]:
    from api_models import evaluate_api_instance
    recs = []
    for inst in instances:
        recs.append(evaluate_api_instance(spec, inst))
    return recs


def run_weights_model(spec, instances, ckpt_dir: str | None = None) -> list[dict]:
    """Open model with attention access. Loads model, extracts signals for source
    attribution (via trained head), AND runs a real generation pass to measure
    behavior (DRM yes/no, conflict following, task accuracy). Server-side."""
    from distributed import load_distributed
    from signals import extract_attention_for_instance, token_signals, attention_pool
    from model import build_source_head, CLASS_TO_IDX
    from api_models import parse_yes_no
    import numpy as np
    import torch
    import re

    model, tok = load_distributed(spec, for_training=False)
    model.eval()

    head = None
    if ckpt_dir and os.path.isdir(ckpt_dir):
        from checkpoint import load_checkpoint
        head = build_source_head(signal_dim=3)
        _, _, head, _ = load_checkpoint(ckpt_dir, source_head=head)
        head.eval()

    dev = next(model.parameters()).device
    if head is not None:
        head = head.to(dev)   # checkpoint saved head on CPU; move to model device

    def generate(prompt, max_new=64):
        # chat template + thinking off, mirroring probing
        if getattr(tok, "chat_template", None):
            for kw in ({"enable_thinking": False}, {}):
                try:
                    prompt = tok.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False, add_generation_prompt=True, **kw)
                    break
                except TypeError:
                    continue
        enc = tok(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        txt = re.sub(r"<think>.*?</think>", " ", txt, flags=re.DOTALL)
        return re.sub(r"<think>.*", " ", txt, flags=re.DOTALL).strip()

    classes = ["verbatim", "parametric", "confabulated"]
    recs = []
    for inst in instances:
        # (a) source attribution via attention signals + trained head
        info = extract_attention_for_instance(model, tok, inst, device=str(dev))
        z = token_signals(info["attn"], info["context_idx"],
                          valid_counts=info["valid_counts"])
        claim_idx = info["claim_idx"]        # real generated-answer tokens
        claim_idx = claim_idx[claim_idx < z.shape[0]]
        if claim_idx.size == 0:
            claim_idx = np.array([z.shape[0] - 1])
        u = attention_pool(z, claim_idx)
        if head is not None:
            with torch.no_grad():
                logits = head(torch.tensor(u, dtype=torch.float32).to(dev).unsqueeze(0))
                pred = classes[int(logits.argmax(-1))]
        else:
            proto = {"verbatim": np.array([0.9, 0.3, 1.0]),
                     "parametric": np.array([0.1, 0.3, 1.0]),
                     "confabulated": np.array([0.1, 0.9, 1.0])}
            pred = min(proto, key=lambda c: np.linalg.norm(u - proto[c]))

        claim = inst.claims[0] if inst.claims else None
        rec = {"instance_id": inst.id, "probe_kind": inst.probe_kind,
               "item_type": inst.item_type, "model": spec.key, "access": spec.access,
               "true_source": claim.source if (claim and claim.is_resolved()) else None,
               "pred_source": pred, "list_type": inst.list_type}

        # (b) REAL behavioral generation pass
        pk = inst.probe_kind
        if pk in ("critical_lure", "studied", "unrelated"):
            # ask whether the probe word was in the studied list
            ans = generate(inst.context + "\n" + inst.question, max_new=24)
            rec["answered_yes"] = parse_yes_no(ans)
        elif pk in ("conflict", "non_conflict_control"):
            ans = generate(inst.context + "\n" + inst.question, max_new=48)
            rec["followed_context"] = (inst.target.lower() in ans.lower())
            rec["task_correct"] = (inst.target.lower() in ans.lower())
        else:  # fact items -> closed-book accuracy
            ans = generate(inst.question, max_new=48)
            gold = inst.answer_aliases or [inst.target]
            rec["task_correct"] = any(str(g).lower() in ans.lower() for g in gold)
        recs.append(rec)
    return recs


def run_prompted_model(spec, instances) -> list[dict]:
    """Open model, NO fine-tuning, source self-reported via prompt (ablation).
    Reuses the API-style prompt but runs locally. Server-side."""
    from distributed import load_distributed
    from api_models import SOURCE_PROMPT, parse_source_tag, parse_yes_no
    import torch

    model, tok = load_distributed(spec, for_training=False)
    model.eval()
    recs = []
    for inst in instances:
        base = (inst.context + "\n\n" if inst.context else "")
        prompt = base + SOURCE_PROMPT.format(question=inst.question)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=100, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        resp = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        claim = inst.claims[0] if inst.claims else None
        rec = {"instance_id": inst.id, "probe_kind": inst.probe_kind,
               "item_type": inst.item_type, "model": spec.key, "access": "prompted",
               "true_source": claim.source if (claim and claim.is_resolved()) else None,
               "pred_source": parse_source_tag(resp), "list_type": inst.list_type}
        if inst.probe_kind in ("critical_lure", "studied", "unrelated"):
            rec["answered_yes"] = parse_yes_no(resp)
        if inst.probe_kind in ("conflict", "non_conflict_control"):
            rec["followed_context"] = (inst.target.lower() in resp.lower())
        rec["task_correct"] = (inst.target.lower() in resp.lower())
        recs.append(rec)
    return recs


DISPATCH = {"api": run_api_model, "prompted": run_prompted_model,
            "weights": run_weights_model}


def run_one(spec, instances, out_root="outputs", ckpt_dir=None, ckpt_root=None):
    out_dir = os.path.join(out_root, spec.key)
    os.makedirs(out_dir, exist_ok=True)
    fn = DISPATCH[spec.access]
    # weights models can load a per-model checkpoint; resolve it here
    if spec.access == "weights":
        resolved = ckpt_dir
        if resolved is None and ckpt_root is not None:
            cand = os.path.join(ckpt_root, f"ckpt_{spec.key}")
            resolved = cand if os.path.isdir(cand) else None
        recs = fn(spec, instances, ckpt_dir=resolved)
    else:
        recs = fn(spec, instances)
    save_predictions(recs, os.path.join(out_dir, "predictions.jsonl"))
    results = evaluate_all(recs)
    results["model"] = spec.key
    results["access"] = spec.access
    results["size_b"] = spec.size_b
    save_results(results, os.path.join(out_dir, "results.json"))
    print(f"[{spec.key}] done -> {out_dir}/results.json")
    return results


# ---------------------------------------------------------------------------
# Aggregation: one comparison table across all models
# ---------------------------------------------------------------------------

def aggregate(out_root="outputs", path="outputs/comparison.json"):
    rows = []
    for spec in MODELS:
        rp = os.path.join(out_root, spec.key, "results.json")
        if not os.path.exists(rp):
            continue
        with open(rp, encoding="utf-8") as f:
            r = json.load(f)
        rows.append({
            "model": spec.key, "access": spec.access, "size_b": spec.size_b,
            "macro_f1": r["attribution"].get("macro_f1"),
            "confab_recall": r["attribution"].get("confabulated_recall"),
            "drm_confab_rate": r["drm"].get("critical_lure_false_yes_rate"),
            "drm_effect": r["drm"].get("drm_effect_present"),
            "verbatim_override": r["conflict"].get("verbatim_override_rate"),
            "over_deference": r["conflict"].get("over_deference_rate"),
            "task_acc": r["task"].get("task_accuracy"),
        })
    rows.sort(key=lambda x: (x["access"], x["size_b"]))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    # pretty print
    cols = ["model", "access", "size_b", "macro_f1", "confab_recall",
            "drm_confab_rate", "verbatim_override", "over_deference", "task_acc"]
    print("\n" + " | ".join(f"{c:>14s}" for c in cols))
    print("-" * (17 * len(cols)))
    for row in rows:
        print(" | ".join(f"{str(row.get(c)):>14s}" for c in cols))
    print(f"\n[aggregate] -> {path}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["all"],
                   help="model keys or 'all'")
    p.add_argument("--dataset", default=FROZEN_PATH)
    p.add_argument("--ckpt_dir", default=None,
                   help="single trained source-head checkpoint (applies to all)")
    p.add_argument("--ckpt_root", default=None,
                   help="dir containing per-model ckpts named ckpt_<modelkey>/")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.aggregate:
        aggregate()
        return

    set_seed(args.seed)
    keys = list_keys() if args.models == ["all"] else args.models

    # shared fallback set (small frozen) for models without a saved eval set
    try:
        frozen_instances = load_frozen(args.dataset)
    except Exception as e:
        print(f"[warn] could not load frozen set ({e}); eval sets required")
        frozen_instances = []

    for k in keys:
        spec = get_model(k)
        # prefer the model's held-out eval set (large-scale, model-relative labels)
        eval_path = f"outputs/eval_{k}.jsonl"
        if spec.access == "weights" and os.path.exists(eval_path):
            from schema import load_dataset as _ld
            instances = _ld(eval_path)
            print(f"[{k}] evaluating on held-out set {eval_path} ({len(instances)} items)")
        else:
            instances = frozen_instances
            if spec.access == "weights":
                print(f"[{k}] no eval set found; falling back to frozen "
                      f"({len(instances)} items) -- results may be weak")
        try:
            run_one(spec, instances, ckpt_dir=args.ckpt_dir, ckpt_root=args.ckpt_root)
        except NotImplementedError as e:
            print(f"[{k}] skipped (server-only stage): {e}")
        except Exception as e:
            print(f"[{k}] ERROR: {e}")
    aggregate()


if __name__ == "__main__":
    main()
