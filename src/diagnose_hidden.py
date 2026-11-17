"""
diagnose_hidden.py  (SERVER-SIDE)
=================================
Path A test: attention signals can't separate parametric vs confabulated. Can the
model's HIDDEN STATES? Several hallucination-detection results suggest a model's
internal representation "knows" when it's fabricating, even when attention and
output entropy don't show it.

Method: for each fact item, generate the answer, take the last-layer hidden state
at the last answer token, and collect these vectors by source label. Then:
  1. report a simple linear separability score (logistic regression, CV accuracy)
     for parametric vs confabulated -- chance is ~0.5; >>0.5 means separable.
  2. compare against a random-feature baseline so we know the score is real.

If hidden states separate the two classes well, it's worth adding a hidden-state
feature to the source head and retraining. If not, path A also fails and we go to
path B (report the finding; measure confabulation behaviorally).

Usage:
    python diagnose_hidden.py --model gemma3-12b --n 400
"""
from __future__ import annotations
import argparse
import numpy as np
import env_load  # noqa
from schema import load_dataset
from registry import get_model
from distributed import load_distributed


def answer_hidden(model, tok, inst, device, max_new=24, layer_frac=1.0):
    """Generate answer; return hidden state at the last answer token, from the
    layer at the given fraction of depth (1.0 = last layer, 0.6 = 60% deep)."""
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
    # hidden_states: tuple(n_layers+1) -- [0]=embeddings, [-1]=final layer.
    hs = out.hidden_states
    n_layers = len(hs) - 1                      # exclude embedding layer
    li = int(round(layer_frac * n_layers))
    li = max(1, min(li, n_layers))              # clamp to [1, n_layers]
    h = hs[li][0].float().cpu().numpy()         # [seq, hidden]
    idx = full.shape[1] - 1
    return h[idx]


def linear_separability(X, y, n_splits=5):
    """CV accuracy of logistic regression; balanced classes downsampled.
    Falls back to a numpy least-squares classifier if sklearn is unavailable."""
    y = np.array(y)
    idx0 = np.where(y == 0)[0]; idx1 = np.where(y == 1)[0]
    m = min(len(idx0), len(idx1))
    if m < 10:
        return None, m
    rng = np.random.RandomState(0)
    sel = np.concatenate([rng.choice(idx0, m, False), rng.choice(idx1, m, False)])
    Xb, yb = X[sel], y[sel]
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, C=1.0))
        scores = cross_val_score(clf, Xb, yb, cv=min(n_splits, m), scoring="accuracy")
        return float(scores.mean()), m
    except ImportError:
        # numpy fallback: standardize, ridge least-squares, k-fold accuracy
        mu, sd = Xb.mean(0), Xb.std(0) + 1e-8
        Xs = (Xb - mu) / sd
        yb2 = yb * 2.0 - 1.0
        k = min(n_splits, m)
        folds = np.array_split(rng.permutation(len(Xs)), k)
        accs = []
        for i in range(k):
            te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
            A = Xs[tr]; b = yb2[tr]
            w = np.linalg.lstsq(A.T @ A + 1e-1 * np.eye(A.shape[1]), A.T @ b, rcond=None)[0]
            pred = (Xs[te] @ w > 0).astype(int)
            accs.append((pred == yb[te]).mean())
        return float(np.mean(accs)), m


def _collect_all_layers(model, tok, inst, device, max_new=24):
    """Generate once; return hidden states at the last answer token for ALL
    layers -> dict layer_index -> vector. Efficient: one generation per item."""
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
    idx = full.shape[1] - 1
    return {li: hs[li][0, idx].float().cpu().numpy() for li in range(len(hs))}


def run(model_key="gemma3-12b", n=400, layer_fracs=(0.4, 0.6, 0.8, 1.0),
        source="eval"):
    spec = get_model(model_key)
    model, tok = load_distributed(spec, for_training=False)
    model.eval()
    dev = next(model.parameters()).device

    data = load_dataset(f"outputs/{'probed' if source=='probed' else 'eval'}_{model_key}.jsonl")
    facts = [x for x in data if x.item_type == "fact" and x.claims
             and x.claims[0].is_resolved()
             and x.claims[0].source in ("parametric", "confabulated")][:n]
    print(f"\n=== hidden-state separability by layer for {model_key} "
          f"({len(facts)} fact items) ===")

    # collect all-layer vectors once per item
    per_layer = {}   # layer_index -> list of vectors
    y = []
    n_layers = None
    for i, inst in enumerate(facts):
        try:
            hd = _collect_all_layers(model, tok, inst, dev)
        except Exception as e:
            print(f"  (skip {inst.id}: {e})"); continue
        if n_layers is None:
            n_layers = len(hd) - 1     # exclude embedding layer 0
        for li, v in hd.items():
            per_layer.setdefault(li, []).append(v)
        y.append(1 if inst.claims[0].source == "confabulated" else 0)
        if (i + 1) % 100 == 0:
            print(f"  collected {i+1}/{len(facts)}")
    y = np.array(y)
    print(f"collected {len(y)} items; parametric={int((y==0).sum())} "
          f"confabulated={int((y==1).sum())}; n_layers={n_layers}\n")

    # which layer indices to report
    want = sorted({max(1, min(n_layers, int(round(f * n_layers))))
                   for f in layer_fracs})
    rng = np.random.RandomState(1)
    yshuf = rng.permutation(y)

    print(f"{'layer':>6s} {'frac':>5s}  {'CV_acc':>7s} {'shuffled':>9s} "
          f"{'net':>6s}  verdict")
    print("-" * 52)
    best = (None, -1)
    for li in want:
        X = np.array(per_layer[li])
        acc, m = linear_separability(X, y)
        acc_r, _ = linear_separability(X, yshuf)
        if acc is None:
            print(f"{li:>6d}   --   (too few minority: {m})"); continue
        net = acc - acc_r
        frac = li / n_layers
        verdict = "SEPARABLE" if (acc >= 0.65 and net >= 0.1) else \
                  ("weak" if net >= 0.05 else "no")
        print(f"{li:>6d} {frac:>5.2f}  {acc:>7.3f} {acc_r:>9.3f} {net:>6.3f}  {verdict}")
        if net > best[1]:
            best = (li, net)
    if best[0] is not None:
        print(f"\nbest layer = {best[0]} (net separability {best[1]:.3f})")
        print("net = CV_acc - shuffled; net>=0.1 with acc>=0.65 = real signal")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma3-12b")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--layers", default="0.4,0.6,0.8,1.0",
                   help="comma-separated depth fractions to test")
    p.add_argument("--source", choices=["eval", "probed"], default="eval")
    args = p.parse_args()
    fracs = tuple(float(x) for x in args.layers.split(","))
    run(args.model, args.n, layer_fracs=fracs, source=args.source)
