"""
distributed.py  (SERVER-SIDE)
=============================
Loading and running LARGE open models sharded across the 4x RTX 5090.

Strategy chosen: one big model split across all 4 GPUs.
  - Inference / signal extraction: device_map="auto" lets HF/accelerate place
    layers across GPUs automatically (pipeline-style sharding). Simple, robust.
  - Training (QLoRA on 30B): 4-bit base sharded via device_map + LoRA adapters.
    For true data-parallel multi-GPU training you'd launch with `accelerate`
    (config notes at the bottom); for a 30B QLoRA fit, device_map sharding on a
    single process is usually the path of least resistance on 4x32GB.

torch/transformers/peft/bitsandbytes imported lazily.

IMPORTANT (repeats the design rule): probe parametric labels with the
NON-quantized model; only the training pass uses 4-bit weights.
"""

from __future__ import annotations
import os
import env_load  # noqa: F401  -- auto-loads .env (HF_TOKEN)


def gpu_report() -> str:
    """Human-readable GPU inventory; safe to call without torch (returns note)."""
    try:
        import torch
    except ImportError:
        return "torch not installed (run on the server)"
    if not torch.cuda.is_available():
        return "CUDA not available"
    lines = [f"{torch.cuda.device_count()} GPU(s):"]
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        lines.append(f"  [{i}] {p.name}  {p.total_memory/1e9:.0f} GB")
    return "\n".join(lines)


def load_distributed(spec, for_training: bool = False, device_map=None):
    """Load a model. Small models go on ONE GPU (cuda:0); only models flagged
    `distributed` are sharded across all GPUs via device_map="auto".

    Sharding an 8B model causes cross-device tensor errors and slow training --
    a single 32GB RTX 5090 holds 8-14B fine. Reserve multi-GPU for 30B+.

    spec: registry.ModelSpec with access=="weights".
    for_training: if True and spec.quantize_4bit, load in 4-bit for QLoRA.
    Returns (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.getenv("HF_TOKEN")  # for gated models (Llama etc.)

    # choose placement: shard only large 'distributed' models; else single GPU
    if device_map is None:
        if getattr(spec, "distributed", False):
            device_map = "auto"                 # shard 30B+ across all GPUs
        else:
            device_map = {"": 0}                # whole model on cuda:0

    # attn_implementation="eager" is REQUIRED: we extract attention weights as
    # source signals, and SDPA/flash attention do not return output_attentions.
    kwargs = dict(torch_dtype=torch.bfloat16, device_map=device_map,
                  output_attentions=True, attn_implementation="eager")
    if hf_token:
        kwargs["token"] = hf_token

    if for_training and spec.quantize_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")

    tok_kwargs = {"token": hf_token} if hf_token else {}
    tok = AutoTokenizer.from_pretrained(spec.hf_name_or_api_id, **tok_kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(spec.hf_name_or_api_id, **kwargs)
    return model, tok


def attach_lora(model, r: int = 8, alpha: int = 16, dropout: float = 0.1,
                target_modules=("q_proj", "v_proj", "o_proj"),
                prepare_kbit: bool = False):
    """Wrap a (sharded, possibly 4-bit) model with LoRA adapters."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if prepare_kbit:
        model = prepare_model_for_kbit_training(model)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=list(target_modules), task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


# ---------------------------------------------------------------------------
# accelerate config notes (write to ~/.cache/huggingface/accelerate/default_config.yaml
# or pass --config_file). For 4x single-node, a minimal multi-GPU config:
#
#   compute_environment: LOCAL_MACHINE
#   distributed_type: MULTI_GPU        # or FSDP for full-shard data parallel
#   num_processes: 4
#   mixed_precision: bf16
#
# Then launch training with:
#   accelerate launch --num_processes 4 train_server.py --model qwen2.5-32b
#
# For a 30B QLoRA fit that won't data-parallelize cleanly, prefer a SINGLE
# process with device_map sharding (no accelerate launch):
#   python train_server.py --model qwen2.5-32b
# ---------------------------------------------------------------------------

def suggest_launch(spec) -> str:
    if spec.distributed and spec.quantize_4bit:
        return ("python train_server.py --model %s   "
                "# single process, device_map shards 4-bit base across 4 GPUs"
                % spec.key)
    if spec.distributed:
        return ("accelerate launch --num_processes 4 train_server.py --model %s"
                % spec.key)
    return "python train_server.py --model %s   # fits on one GPU" % spec.key


if __name__ == "__main__":
    print(gpu_report())
