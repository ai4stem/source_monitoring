"""
extract_hidden.py  (SERVER-SIDE)
================================
Extract hidden states ONCE per model and save them, so any downstream probe
(linear, MLP, calibration) can reuse them without re-running generation.

For each fact item in the model's probed pool we generate the answer, take the
last-layer-swept hidden state at the answer token for a set of depth fractions,
and save a compact .npz:

    outputs/hidden_<model>.npz
        X_<frac>   float16 [n, hidden]   hidden states at that depth fraction
        y          int8    [n]           1=confabulated, 0=parametric
        fracs      float   [k]           the depth fractions saved
        sizes      ...                   bookkeeping

This is the expensive step (needs generation). Run it once; then
probe_hidden.py does linear/MLP/calibration on the saved arrays in seconds.

Usage:
    python extract_hidden.py --model qwen3-8b --n 2000
    python extract_hidden.py --model all --n 2000
"""
from __future__ import annotations
import argparse, os
import numpy as np
import env_load  # noqa
from schema import load_dataset
from registry import get_model, MODELS
from distributed import load_distributed


def collect(model, tok, inst, device, fracs, max_new=24):
    """Generate answer; return {frac: hidden_vec} at the last answer token."""
    import torch
    user = inst.question
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
    enc = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    full = gen[0].unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=full, output_hidden_states=True)
    hs = out.hidden_states
    n_layers = len(hs) - 1
    idx = full.shape[1] - 1
    result = {}
    for f in fracs:
        li = max(1, min(n_layers, int(round(f * n_layers))))
        v = hs[li][0, idx].float().cpu().numpy()
        # clip to float16 range so large activations (e.g. gemma) don't become
        # inf on save; probes standardize anyway so clipping is harmless.
        v = np.clip(v, -65000, 65000).astype(np.float16)
        result[f] = v
    return result


def run(model_key, n=2000, fracs=(0.4, 0.6, 0.8, 1.0), source="probed"):
    spec = get_model(model_key)
    path = f"outputs/{'probed' if source=='probed' else 'eval'}_{model_key}.jsonl"
    if not os.path.exists(path):
        print(f"[skip] {model_key}: no {source} set")
        return
    model, tok = load_distributed(spec, for_training=False)
    model.eval()
    dev = next(model.parameters()).device

    data = load_dataset(path)
    facts = [x for x in data if x.item_type == "fact" and x.claims
             and x.claims[0].is_resolved()
             and x.claims[0].source in ("parametric", "confabulated")][:n]
    print(f"[extract] {model_key}: {len(facts)} fact items")

    buckets = {f: [] for f in fracs}
    y = []
    for i, inst in enumerate(facts):
        try:
            h = collect(model, tok, inst, dev, fracs)
        except Exception as e:
            print(f"  (skip {inst.id}: {e})"); continue
        for f in fracs:
            buckets[f].append(h[f])
        y.append(1 if inst.claims[0].source == "confabulated" else 0)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(facts)}")

    y = np.array(y, dtype=np.int8)
    out = {f"X_{f}": np.array(buckets[f], dtype=np.float16) for f in fracs}
    out["y"] = y
    out["fracs"] = np.array(fracs, dtype=float)
    os.makedirs("outputs", exist_ok=True)
    fn = f"outputs/hidden_{model_key}.npz"
    np.savez_compressed(fn, **out)
    print(f"[extract] saved {fn}  (n={len(y)}, "
          f"param={int((y==0).sum())}, confab={int((y==1).sum())})")

    del model
    import torch, gc
    gc.collect(); torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3-8b")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--source", choices=["eval", "probed"], default="probed")
    p.add_argument("--layers", default="0.4,0.6,0.8,1.0")
    args = p.parse_args()
    fracs = tuple(float(x) for x in args.layers.split(","))
    keys = [m.key for m in MODELS if m.access == "weights"] \
        if args.model == "all" else [args.model]
    for k in keys:
        run(k, n=args.n, fracs=fracs, source=args.source)


if __name__ == "__main__":
    main()
