"""
measure_drm.py  (SERVER-SIDE)
=============================
DRM false-memory measurement, done the RIGHT way: pure behavioral generation.

DRM confabulation is NOT a signal/head judgement -- it's whether the model, shown
a studied word list, then FALSELY says the (non-presented) critical lure "was in
the list". So we just ask and parse yes/no. No attention, no source head.

For each DRM list we have three probes:
  - critical_lure : lure word (NOT in list) -> a "yes" is a FALSE MEMORY
  - studied       : a real list word        -> "yes" is correct (true recall)
  - unrelated     : unrelated word          -> "yes" is a plain false alarm

Headline numbers (per model):
  false_recall_rate   = P(yes | critical_lure)      -- the DRM illusion
  true_recall_rate    = P(yes | studied)            -- positive control
  base_false_alarm    = P(yes | unrelated)          -- negative control
  drm_effect          = false_recall_rate - base_false_alarm  (the real effect)

A genuine DRM effect: false_recall_rate >> base_false_alarm (model is fooled by
semantic gist, not just guessing yes).

Usage:
    python measure_drm.py --model gemma3-12b
    python measure_drm.py --model all
"""
from __future__ import annotations
import argparse, json, os
import env_load  # noqa
from build_dataset import build_drm_items
from registry import get_model, list_keys
from distributed import load_distributed
from api_models import parse_yes_no


def measure_one(model_key, seed=0, max_new=16, verbose=False):
    spec = get_model(model_key)
    if spec.access != "weights":
        raise NotImplementedError(f"{model_key} is not a weights model")
    model, tok = load_distributed(spec, for_training=False)
    model.eval()
    dev = next(model.parameters()).device
    import torch

    def ask(context, question):
        user = f"{context}\n{question}\nAnswer yes or no."
        prompt = user
        if getattr(tok, "chat_template", None):
            for kw in ({"enable_thinking": False}, {}):
                try:
                    prompt = tok.apply_chat_template(
                        [{"role": "user", "content": user}],
                        tokenize=False, add_generation_prompt=True, **kw)
                    break
                except TypeError:
                    continue
        enc = tok(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        txt = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        import re
        txt = re.sub(r"<think>.*?</think>", " ", txt, flags=re.DOTALL)
        txt = re.sub(r"<think>.*", " ", txt, flags=re.DOTALL).strip()
        return parse_yes_no(txt)   # True/False/None

    items = build_drm_items(seed=seed)
    tally = {"critical_lure": [], "studied": [], "unrelated": []}
    for inst in items:
        pk = inst.probe_kind
        if pk not in tally:
            continue
        yes = ask(inst.context, inst.question)
        if yes is not None:
            tally[pk].append(1 if yes else 0)
        if verbose:
            print(f"  [{pk}] lure={inst.lure} -> {'YES' if yes else 'no'}")

    def rate(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    fr = rate(tally["critical_lure"])
    tr = rate(tally["studied"])
    ba = rate(tally["unrelated"])
    eff = round(fr - ba, 4) if (fr is not None and ba is not None) else None
    return {
        "model": model_key, "size_b": spec.size_b,
        "false_recall_rate": fr,      # P(yes | critical lure) = DRM illusion
        "true_recall_rate": tr,       # P(yes | studied)       = positive control
        "base_false_alarm": ba,       # P(yes | unrelated)     = negative control
        "drm_effect": eff,            # false_recall - base_false_alarm
        "n_lure": len(tally["critical_lure"]),
        "n_studied": len(tally["studied"]),
        "n_unrelated": len(tally["unrelated"]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma3-12b")
    p.add_argument("--out", default="outputs/drm_results.json")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    keys = [k for k in list_keys() if get_model(k).access == "weights"] \
        if args.model == "all" else [args.model]

    results = []
    for k in keys:
        print(f"\n=== measuring DRM: {k} ===")
        try:
            r = measure_one(k, verbose=args.verbose)
            results.append(r)
            print(f"  false_recall={r['false_recall_rate']} "
                  f"true_recall={r['true_recall_rate']} "
                  f"base_false_alarm={r['base_false_alarm']} "
                  f"drm_effect={r['drm_effect']}")
        except Exception as e:
            print(f"  ERROR: {e}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {args.out}")

    # summary table
    print(f"\n{'model':18s} {'size':>5s} {'false_recall':>13s} "
          f"{'true_recall':>12s} {'base_FA':>8s} {'drm_effect':>11s}")
    print("-" * 72)
    for r in results:
        print(f"{r['model']:18s} {r['size_b']:>5.1f} "
              f"{str(r['false_recall_rate']):>13s} {str(r['true_recall_rate']):>12s} "
              f"{str(r['base_false_alarm']):>8s} {str(r['drm_effect']):>11s}")


if __name__ == "__main__":
    main()
