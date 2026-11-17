"""
diagnose_signals.py  (SERVER-SIDE)
==================================
The key question before any retraining: do the attention signals over the
GENERATED ANSWER tokens actually separate the three sources? Especially the
hard pair, parametric vs confabulated.

This runs the FIXED extract_attention_for_instance (which now generates the
answer and measures attention over the answer span) on a model's eval set, and
prints the mean signal per source class -- with NO trained head involved. If the
per-class means are well separated, retraining is worth it; if parametric and
confabulated overlap, attention signals alone can't do 3-way and we need a
different approach.

Usage:
    python diagnose_signals.py --model gemma3-12b --n 120
"""
from __future__ import annotations
import argparse
import numpy as np
import env_load  # noqa
from schema import load_dataset
from registry import get_model
from distributed import load_distributed
from signals import extract_attention_for_instance, token_signals, attention_pool


def run(model_key="gemma3-12b", n=120):
    spec = get_model(model_key)
    model, tok = load_distributed(spec, for_training=False)
    model.eval()
    dev = next(model.parameters()).device

    path = f"outputs/eval_{model_key}.jsonl"
    data = load_dataset(path)[:n]
    print(f"\n=== signal separation for {model_key} "
          f"({len(data)} eval items) ===")
    print("signal u = [context_mass m, normalized_entropy H~, consensus c]\n")

    # IMPORTANT: separate fact items from DRM items. DRM probes have a context
    # (the word list) so their m is high, while closed-book facts have m=0.
    # Mixing them in one 'confabulated' bucket pollutes the signal. We bucket by
    # (source, has_context) so the parametric-vs-confabulated question is asked
    # ONLY among no-context fact items, which is the fair comparison.
    by_src = {"verbatim": [], "parametric": [], "confabulated": [],
              "drm_confab": []}
    for inst in data:
        if not inst.claims or not inst.claims[0].is_resolved():
            continue
        src = inst.claims[0].source
        is_drm = inst.item_type == "drm"
        try:
            info = extract_attention_for_instance(model, tok, inst, device=str(dev))
            z = token_signals(info["attn"], info["context_idx"],
                              valid_counts=info["valid_counts"])
            cidx = info["claim_idx"]
            cidx = cidx[cidx < z.shape[0]]
            if cidx.size == 0:
                cidx = np.array([z.shape[0] - 1])
            u = attention_pool(z, cidx)
        except Exception as e:
            print(f"  (skip {inst.id}: {e})")
            continue
        if is_drm:
            if src == "confabulated":
                by_src["drm_confab"].append(u)
        elif src in ("verbatim", "parametric", "confabulated"):
            by_src[src].append(u)

    print(f"{'source':14s} {'n':>4s}  {'m':>8s} {'H~':>8s} {'c':>8s}")
    print("-" * 48)
    means = {}
    for src, us in by_src.items():
        if not us:
            print(f"{src:14s} {0:>4d}  (none)")
            continue
        arr = np.array(us)
        mu = arr.mean(axis=0)
        sd = arr.std(axis=0)
        means[src] = mu
        print(f"{src:14s} {len(us):>4d}  "
              f"{mu[0]:.3f}±{sd[0]:.2f} {mu[1]:.3f}±{sd[1]:.2f} {mu[2]:.3f}±{sd[2]:.2f}")

    # the crucial contrast
    if "parametric" in means and "confabulated" in means:
        d = np.abs(means["parametric"] - means["confabulated"])
        print("\nparametric vs confabulated |Δ| per signal:")
        print(f"   Δm={d[0]:.3f}  ΔH~={d[1]:.3f}  Δc={d[2]:.3f}")
        print("   (large Δ on some signal = separable; all small = not separable "
              "by attention alone)")
    if "verbatim" in means and "parametric" in means:
        d = np.abs(means["verbatim"] - means["parametric"])
        print(f"verbatim vs parametric |Δ|: Δm={d[0]:.3f} ΔH~={d[1]:.3f} Δc={d[2]:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma3-12b")
    p.add_argument("--n", type=int, default=120)
    args = p.parse_args()
    run(args.model, args.n)
