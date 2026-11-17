"""
registry.py
===========
Declarative registry of every model to compare, with its *access mode*.

Three access modes determine how source signals are obtained:
  - "weights"   : open model, full attention access -> our intrinsic signals
                  (context mass m, entropy H~) + optional fine-tuning.
  - "prompted"  : open model used WITHOUT fine-tuning, source self-reported via
                  prompt (ablation: what prompting alone achieves).
  - "api"       : commercial LLM (GPT-4o / Claude). Weights inaccessible, so
                  source is self-reported via prompt design ONLY. Cannot produce
                  attention-based signals -- compared on behavioral metrics
                  (DRM confab rate, conflict resolution, self-reported source).

`distributed` flags large models that must be sharded across the 4 GPUs.
Sizes span small->large so the scale axis (does confabulation persist?) is testable.

Edit MODELS to match what is current/available at run time. Model identifiers
go stale quickly -- verify on the hub before a run.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional

AccessMode = Literal["weights", "prompted", "api"]


@dataclass
class ModelSpec:
    key: str                              # short id used in result filenames
    hf_name_or_api_id: str
    access: AccessMode
    size_b: float                         # parameter count in billions (approx)
    distributed: bool = False             # shard across GPUs (large models)
    quantize_4bit: bool = False           # QLoRA / 4-bit load (large open models)
    api_provider: Optional[str] = None    # "openai" | "anthropic"
    notes: str = ""


# ---------------------------------------------------------------------------
# EDIT THIS LIST. Identifiers reflect the June 2026 landscape. Model strings go
# stale every ~6 weeks -- ALWAYS confirm the exact current id on HuggingFace /
# provider docs before a run (the keys below are the registry's stable handle;
# only hf_name_or_api_id changes when you bump a version).
#
# Tiering rationale:
#   primary  : current small open models -> "even the newest models still
#              confabulate" (the claim that defends novelty in 2026).
#   legacy   : the prior-paper models kept as a generational control -> lets you
#              show source-monitoring deficits PERSIST across generations and
#              compare directly to your Sci Rep results.
#   scale    : a large current open model sharded across the 4x5090.
#   api      : current commercial models, prompt-design only.
# ---------------------------------------------------------------------------
MODELS: list[ModelSpec] = [
    # --- primary: current small open models (June 2026) ---
    ModelSpec("qwen3-8b", "Qwen/Qwen3-8B",
              access="weights", size_b=8.0, notes="primary; Apache-2.0"),
    ModelSpec("gemma3-12b", "google/gemma-3-12b-it",
              access="weights", size_b=12.0, notes="primary; verify exact id"),
    ModelSpec("phi4-mini", "microsoft/Phi-4-mini-instruct",
              access="weights", size_b=3.8, notes="primary; MIT, fits small VRAM"),

    # --- legacy: prior-paper models as a generational control ---
    ModelSpec("llama3-8b-legacy", "meta-llama/Meta-Llama-3-8B-Instruct",
              access="weights", size_b=8.0,
              notes="generational control; matches prior Sci Rep paper"),
    ModelSpec("phi3-mini-legacy", "microsoft/Phi-3-mini-4k-instruct",
              access="weights", size_b=3.8,
              notes="generational control; matches prior Sci Rep paper"),

    # --- prompted ablation (current primary model, no fine-tuning) ---
    ModelSpec("qwen3-8b-prompted", "Qwen/Qwen3-8B",
              access="prompted", size_b=8.0, notes="prompt-only ablation"),

    # --- scale axis: large current open model sharded across 4x RTX 5090 ---
    ModelSpec("qwen3-32b", "Qwen/Qwen3-32B",
              access="weights", size_b=32.0, distributed=True, quantize_4bit=True,
              notes="scale axis; QLoRA train, full-precision probe; verify id"),

    # --- SIZE LADDER: same architecture across sizes, to test whether the
    #     hidden-state separability of confabulation SCALES with model size.
    #     Qwen3 ladder (with qwen3-8b, qwen3-32b above -> 1.7/4/8/14/32B).
    ModelSpec("qwen3-1.7b", "Qwen/Qwen3-1.7B",
              access="weights", size_b=1.7, notes="Qwen3 size ladder; Apache-2.0"),
    ModelSpec("qwen3-4b", "Qwen/Qwen3-4B",
              access="weights", size_b=4.0, notes="Qwen3 size ladder; Apache-2.0"),
    ModelSpec("qwen3-14b", "Qwen/Qwen3-14B",
              access="weights", size_b=14.0, notes="Qwen3 size ladder; Apache-2.0"),
    #     Llama-3.2 ladder (with llama3-8b-legacy -> 1/3/8B, different arch).
    ModelSpec("llama32-1b", "meta-llama/Llama-3.2-1B-Instruct",
              access="weights", size_b=1.0, notes="Llama size ladder"),
    ModelSpec("llama32-3b", "meta-llama/Llama-3.2-3B-Instruct",
              access="weights", size_b=3.0, notes="Llama size ladder"),

    # --- ARCHITECTURE diversity: Mistral family (different arch from Qwen/Llama/
    #     Gemma/Phi), to check the finding isn't architecture-specific.
    ModelSpec("mistral-7b", "mistralai/Mistral-7B-Instruct-v0.3",
              access="weights", size_b=7.0, notes="architecture diversity; Apache-2.0"),
    ModelSpec("mistral-nemo-12b", "mistralai/Mistral-Nemo-Instruct-2407",
              access="weights", size_b=12.0,
              notes="architecture diversity; 12B, Apache-2.0"),

    # --- commercial APIs (prompt-design only; no attention signals) ---
    # Confirm the exact current model ids before running -- these change often.
    ModelSpec("gpt-5", "gpt-5.2", access="api", size_b=0.0,
              api_provider="openai", notes="confirm current GPT id at run time"),
    ModelSpec("claude-opus", "claude-opus-4-8", access="api", size_b=0.0,
              api_provider="anthropic",
              notes="Claude Opus 4.8 (current GA flagship as of 2026-06)"),
    ModelSpec("claude-sonnet", "claude-sonnet-4-6", access="api", size_b=0.0,
              api_provider="anthropic",
              notes="Claude Sonnet 4.6; cheaper API comparison point"),
]


def get_model(key: str) -> ModelSpec:
    for m in MODELS:
        if m.key == key:
            return m
    raise KeyError(f"unknown model key: {key}")


def list_keys() -> list[str]:
    return [m.key for m in MODELS]


def by_access(mode: AccessMode) -> list[ModelSpec]:
    return [m for m in MODELS if m.access == mode]


def primary_open() -> list[ModelSpec]:
    """Current small open models -- the main results."""
    return [m for m in MODELS if m.access == "weights"
            and not m.distributed and "legacy" not in m.key]


def legacy_models() -> list[ModelSpec]:
    """Prior-paper models kept as a generational control."""
    return [m for m in MODELS if "legacy" in m.key]


def size_ladder(family: str = None) -> list[ModelSpec]:
    """Weights models sorted by size, for scaling analysis. Optionally filter to
    a family prefix ('qwen3', 'llama', 'mistral')."""
    ms = [m for m in MODELS if m.access == "weights"]
    if family:
        ms = [m for m in ms if m.key.startswith(family)]
    return sorted(ms, key=lambda m: m.size_b)


if __name__ == "__main__":
    print("Registered models (June 2026):")
    for m in MODELS:
        tag = m.access + ("/dist" if m.distributed else "")
        note = f"  -- {m.notes}" if m.notes else ""
        print(f"  {m.key:20s} {m.size_b:5.1f}B  [{tag:14s}]  {m.hf_name_or_api_id}{note}")
