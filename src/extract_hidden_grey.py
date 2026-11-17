"""
extract_hidden_grey.py  (SERVER-SIDE)  —  EXPERIMENT 1
======================================================
Like extract_hidden.py, but ALSO extracts the grey-zone (prompt_sensitive) items
that the main analysis excludes. This lets us test the following concern:

    "Excluding the grey zone may guarantee the weak-signal result by design.
     The items with a live uncertainty signal are exactly the ones removed."

If the hidden state cleanly separates grey (uncertain) from parametric+confabulated
(stable), while FAILING to separate confabulated from parametric within the stable
set, that is a much stronger and more precise finding: the model DOES encode
'am I sure?' but NOT 'is my stable answer true?'.

Requires the REVISED probing so grey items carry a stability record. Grey items in
probed_<model>.jsonl have claims[0].source == 'pending' (labeled False). We read
them by probe_kind/stability rather than is_resolved().

Saves outputs/hidden3_<model>.npz with:
    X_<frac>  float16 [n, hidden]
    y3        int8    [n]   0=parametric, 1=confabulated, 2=grey
Usage:
    python extract_hidden_grey.py --model qwen3-8b --n 3000
    python extract_hidden_grey.py --model all --n 3000
"""
from __future__ import annotations
import argparse, os
import numpy as np
import env_load  # noqa
from schema import load_dataset
from registry import get_model, MODELS
from distributed import load_distributed


def _label3(inst):
    """0=parametric, 1=confabulated, 2=grey(prompt_sensitive); None=skip."""
    if inst.item_type != "fact" or not inst.claims:
        return None
    c = inst.claims[0]
    if c.source == "parametric":
        return 0
    if c.source == "confabulated":
        return 1
    # grey zone: pending/unlabeled, but must have been probed (has stability)
    stab = getattr(inst, "stability", None) or {}
    if c.source == "pending" or stab.get("category") == "prompt_sensitive":
        return 2
    return None


def collect(model, tok, inst, device, fracs, max_new=24):
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
        v = np.clip(v, -65000, 65000).astype(np.float16)
        result[f] = v
    return result


def run(model_key, n=3000, fracs=(0.4, 0.6, 0.8, 1.0)):
    spec = get_model(model_key)
    path = f"outputs/probed_{model_key}.jsonl"
    if not os.path.exists(path):
        print(f"[skip] {model_key}: no probed set"); return
    data = load_dataset(path)
    items = [(x, _label3(x)) for x in data]
    items = [(x, y) for x, y in items if y is not None][:n]
    if not items:
        print(f"[skip] {model_key}: no labelled items"); return
    from collections import Counter
    dist = Counter(y for _, y in items)
    print(f"[extract3] {model_key}: {len(items)} items "
          f"(param={dist[0]}, confab={dist[1]}, grey={dist[2]})")
    if dist[2] == 0:
        print(f"  WARNING: no grey items found. Re-run probing with REVISED "
              f"probe_knowledge.py so grey items are cached.")

    model, tok = load_distributed(spec, for_training=False)
    model.eval()
    dev = next(model.parameters()).device

    buckets = {f: [] for f in fracs}; y3 = []
    for i, (inst, lab) in enumerate(items):
        try:
            h = collect(model, tok, inst, dev, fracs)
        except Exception as e:
            print(f"  (skip {inst.id}: {e})"); continue
        for f in fracs:
            buckets[f].append(h[f])
        y3.append(lab)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(items)}")

    out = {f"X_{f}": np.array(buckets[f], dtype=np.float16) for f in fracs}
    out["y3"] = np.array(y3, dtype=np.int8)
    out["fracs"] = np.array(fracs, dtype=float)
    os.makedirs("outputs", exist_ok=True)
    fn = f"outputs/hidden3_{model_key}.npz"
    np.savez_compressed(fn, **out)
    from collections import Counter as C
    d = C(y3)
    print(f"[extract3] saved {fn} (param={d[0]}, confab={d[1]}, grey={d[2]})")
    del model
    import torch, gc; gc.collect(); torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3-8b")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--layers", default="0.4,0.6,0.8,1.0")
    args = p.parse_args()
    fracs = tuple(float(x) for x in args.layers.split(","))
    keys = [m.key for m in MODELS if m.access == "weights"] \
        if args.model == "all" else [args.model]
    for k in keys:
        run(k, n=args.n, fracs=fracs)


if __name__ == "__main__":
    main()
