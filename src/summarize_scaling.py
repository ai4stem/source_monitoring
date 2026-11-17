"""
summarize_scaling.py  (SERVER-SIDE)
===================================
Aggregate the hidden-state separability finding ACROSS models, sorted by size,
to answer the scaling question: does the detectability of confabulation (in the
model's internal representation) grow with model size or architecture?

For every weights model that has an eval set (outputs/eval_<key>.jsonl), this
runs the hidden-state separability probe (best layer) and prints one row per
model, sorted by size. It also groups by family so you can see per-architecture
trends (Qwen ladder, Llama ladder, Mistral, etc.).

Reuses diagnose_hidden.run_layers() so the measurement is identical to the
standalone diagnostic.

Usage:
    python summarize_scaling.py                 # all models with eval sets
    python summarize_scaling.py --n 400
    python summarize_scaling.py --out outputs/scaling.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import env_load  # noqa
from registry import MODELS, get_model
from schema import load_dataset
from distributed import load_distributed
from diagnose_hidden import _collect_all_layers, linear_separability


def best_layer_separability(model_key, n=400, layer_fracs=(0.4, 0.6, 0.8, 1.0),
                            source="eval"):
    """Return best-layer separability for a model. source='eval' uses the small
    held-out set; source='probed' uses the full probed pool (many more
    confabulated items -> far more stable estimate)."""
    spec = get_model(model_key)
    path = f"outputs/{'probed' if source == 'probed' else 'eval'}_{model_key}.jsonl"
    if not os.path.exists(path):
        return None
    model, tok = load_distributed(spec, for_training=False)
    model.eval()
    dev = next(model.parameters()).device

    data = load_dataset(path)
    facts = [x for x in data if x.item_type == "fact" and x.claims
             and x.claims[0].is_resolved()
             and x.claims[0].source in ("parametric", "confabulated")][:n]
    per_layer, y, n_layers = {}, [], None
    for inst in facts:
        try:
            hd = _collect_all_layers(model, tok, inst, dev)
        except Exception:
            continue
        if n_layers is None:
            n_layers = len(hd) - 1
        for li, v in hd.items():
            per_layer.setdefault(li, []).append(v)
        y.append(1 if inst.claims[0].source == "confabulated" else 0)
    y = np.array(y)

    # free GPU for the next model
    del model
    import torch, gc
    gc.collect(); torch.cuda.empty_cache()

    if n_layers is None or (y == 1).sum() < 10 or (y == 0).sum() < 10:
        return {"model": model_key, "size_b": spec.size_b, "best_net": None,
                "best_frac": None, "n_param": int((y == 0).sum()),
                "n_confab": int((y == 1).sum()), "note": "too few items"}

    rng = np.random.RandomState(1)
    yshuf = rng.permutation(y)
    want = sorted({max(1, min(n_layers, int(round(f * n_layers))))
                   for f in layer_fracs})
    best = {"net": -1.0, "frac": None, "acc": None}
    for li in want:
        X = np.array(per_layer[li])
        acc, _ = linear_separability(X, y)
        acc_r, _ = linear_separability(X, yshuf)
        if acc is None:
            continue
        net = acc - acc_r
        if net > best["net"]:
            best = {"net": net, "frac": round(li / n_layers, 2), "acc": acc}
    return {"model": model_key, "size_b": spec.size_b,
            "best_net": round(best["net"], 3), "best_frac": best["frac"],
            "best_acc": round(best["acc"], 3) if best["acc"] else None,
            "n_param": int((y == 0).sum()), "n_confab": int((y == 1).sum())}


def family_of(key):
    for fam in ("qwen3", "llama", "mistral", "gemma", "phi"):
        if key.startswith(fam):
            return fam
    return "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--source", choices=["eval", "probed"], default="eval",
                   help="'probed' uses the full 6000-item pool (many more "
                        "confabulated -> more stable); 'eval' uses held-out set")
    p.add_argument("--out", default="outputs/scaling.json")
    p.add_argument("--models", nargs="*", default=None,
                   help="specific keys; default = all weights models with eval sets")
    args = p.parse_args()

    keys = args.models or [m.key for m in MODELS if m.access == "weights"]
    rows = []
    for k in keys:
        fname = f"outputs/{'probed' if args.source == 'probed' else 'eval'}_{k}.jsonl"
        if not os.path.exists(fname):
            print(f"[skip] {k}: no {args.source} set ({fname})")
            continue
        print(f"[measure] {k} (source={args.source}) ...")
        try:
            r = best_layer_separability(k, n=args.n, source=args.source)
            if r:
                r["family"] = family_of(k)
                rows.append(r)
                print(f"   size={r['size_b']}B best_net={r['best_net']} "
                      f"(layer_frac={r['best_frac']}, acc={r.get('best_acc')}) "
                      f"n_confab={r['n_confab']}")
        except Exception as e:
            print(f"   ERROR: {e}")

    rows.sort(key=lambda r: r["size_b"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)

    # table sorted by size
    print(f"\n{'model':18s} {'family':8s} {'size':>5s} {'best_net':>9s} "
          f"{'layer':>6s} {'acc':>6s} {'n_conf':>7s}")
    print("-" * 66)
    for r in rows:
        print(f"{r['model']:18s} {r['family']:8s} {r['size_b']:>5.1f} "
              f"{str(r['best_net']):>9s} {str(r['best_frac']):>6s} "
              f"{str(r.get('best_acc')):>6s} {r['n_confab']:>7d}")

    # per-family trend (size vs net) -- the headline scaling signal
    print("\n=== per-family: size -> best_net (is it rising with size?) ===")
    fams = {}
    for r in rows:
        if r["best_net"] is not None:
            fams.setdefault(r["family"], []).append((r["size_b"], r["best_net"]))
    for fam, pts in sorted(fams.items()):
        pts.sort()
        trail = "  ".join(f"{s:.0f}B:{net:.2f}" for s, net in pts)
        print(f"  {fam:8s} {trail}")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
