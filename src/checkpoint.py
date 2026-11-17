"""
checkpoint.py
=============
Save / load the trainable pieces together:
  - LoRA adapter weights (via peft)
  - SourceHead state_dict (classifier + pooling scorer)
  - learnable head weights phi_h
  - run config (hyperparameters, class weights, model name)

Layout on disk:
  <ckpt_dir>/
      adapter/            # peft save_pretrained
      source_head.pt      # torch state_dict
      config.json         # run config + class weights + metrics pointer

torch/peft imports are lazy so config save/load works without them.
"""

from __future__ import annotations
import json
import os


def save_config(config: dict, ckpt_dir: str) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_config(ckpt_dir: str) -> dict:
    with open(os.path.join(ckpt_dir, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(ckpt_dir: str, peft_model=None, source_head=None,
                    head_weights=None, config: dict | None = None) -> None:
    """Persist all trainable pieces.  torch/peft only needed if those args given."""
    os.makedirs(ckpt_dir, exist_ok=True)
    if config is not None:
        save_config(config, ckpt_dir)
    if peft_model is not None:
        peft_model.save_pretrained(os.path.join(ckpt_dir, "adapter"))
    if source_head is not None:
        import torch
        payload = {"source_head": source_head.state_dict()}
        if head_weights is not None:
            payload["head_weights"] = head_weights.detach().cpu()
        torch.save(payload, os.path.join(ckpt_dir, "source_head.pt"))


def load_checkpoint(ckpt_dir: str, base_model=None, source_head=None,
                    map_location: str = "cpu"):
    """Load config always; load weights if the corresponding objects are passed.

    Returns (config, peft_model_or_None, source_head_or_None, head_weights_or_None).
    """
    config = load_config(ckpt_dir)
    peft_model = None
    head_weights = None

    adapter_dir = os.path.join(ckpt_dir, "adapter")
    if base_model is not None and os.path.isdir(adapter_dir):
        from peft import PeftModel
        peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

    head_path = os.path.join(ckpt_dir, "source_head.pt")
    if source_head is not None and os.path.exists(head_path):
        import torch
        payload = torch.load(head_path, map_location=map_location)
        source_head.load_state_dict(payload["source_head"])
        head_weights = payload.get("head_weights")

    return config, peft_model, source_head, head_weights
