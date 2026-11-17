"""
extract_hidden_uhah.py  (SERVER-SIDE)  —  companion to split_uh_ah.py
=====================================================================
Extracts last-token hidden states for parametric + confabulated facts, storing
ALSO each fact's answer_entropy (from the revised probing stability record) and
its label. This lets split_uh_ah.py divide 'confabulated' into AH (low entropy,
consistent wrong = association-driven) vs UH (high entropy, scattered wrong =
ungrounded), following Cheang et al.'s taxonomy by an independent route.

Saves outputs/hidden_uhah_<model>.npz with:
    X_<frac>  float16 [n, hidden]
    label     int8    [n]   0=parametric, 1=confabulated
    ent       float32 [n]   answer_entropy from probing
    fracs     float   [k]
Usage:
    python extract_hidden_uhah.py --model qwen3-8b --n 4000
    python extract_hidden_uhah.py --model all --n 4000
"""
from __future__ import annotations
import argparse, os
import numpy as np
import env_load  # noqa
from schema import load_dataset
from registry import get_model, MODELS
from distributed import load_distributed


def _label_ent(inst):
    """Return (label, entropy) for param/confab facts; None to skip."""
    if inst.item_type != "fact" or not inst.claims:
        return None
    c = inst.claims[0]
    if not c.is_resolved() or c.source not in ("parametric", "confabulated"):
        return None
    stab = getattr(inst, "stability", None) or {}
    ent = stab.get("answer_entropy", None)
    if ent is None:
        return None
    return (0 if c.source == "parametric" else 1), float(ent)


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
        result[f] = np.clip(v, -65000, 65000).astype(np.float16)
    return result


def run(model_key, n=4000, fracs=(0.4, 0.6, 0.8, 1.0)):
    spec = get_model(model_key)
    path = f"outputs/probed_{model_key}.jsonl"
    if not os.path.exists(path):
        print(f"[skip] {model_key}: no probed set"); return
    data = load_dataset(path)
    items = []
    for x in data:
        le = _label_ent(x)
        if le is not None:
            items.append((x, le[0], le[1]))
    items = items[:n]
    if not items:
        print(f"[skip] {model_key}: no param/confab with entropy"); return
    from collections import Counter
    dist = Counter(lab for _, lab, _ in items)
    print(f"[uhah] {model_key}: {len(items)} items (param={dist[0]}, confab={dist[1]})")

    model, tok = load_distributed(spec, for_training=False)
    model.eval()
    dev = next(model.parameters()).device

    buckets = {f: [] for f in fracs}; labels = []; ents = []
    for i, (inst, lab, ent) in enumerate(items):
        try:
            h = collect(model, tok, inst, dev, fracs)
        except Exception as e:
            print(f"  (skip {inst.id}: {e})"); continue
        for f in fracs:
            buckets[f].append(h[f])
        labels.append(lab); ents.append(ent)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(items)}")

    out = {f"X_{f}": np.array(buckets[f], dtype=np.float16) for f in fracs}
    out["label"] = np.array(labels, dtype=np.int8)
    out["ent"] = np.array(ents, dtype=np.float32)
    out["fracs"] = np.array(fracs, dtype=float)
    os.makedirs("outputs", exist_ok=True)
    fn = f"outputs/hidden_uhah_{model_key}.npz"
    np.savez_compressed(fn, **out)
    print(f"[uhah] saved {fn} (n={len(labels)})")
    del model
    import torch, gc; gc.collect(); torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3-8b")
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--layers", default="0.4,0.6,0.8,1.0")
    args = p.parse_args()
    fracs = tuple(float(x) for x in args.layers.split(","))
    keys = [m.key for m in MODELS if m.access == "weights"] \
        if args.model == "all" else [args.model]
    for k in keys:
        run(k, n=args.n, fracs=fracs)


if __name__ == "__main__":
    main()
