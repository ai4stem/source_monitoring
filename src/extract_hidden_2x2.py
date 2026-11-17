"""
extract_hidden_2x2.py  (SERVER-SIDE)  —  support x truth design
===========================================================================
Extracts last-token hidden states for facts, storing FOR EACH FACT:
  - support : leave-one-paraphrase-out (LOPO) agreement, correctness-independent
  - correct : whether the modal answer is correct (truth)
  - entropy : answer entropy (kept for familiarity/robustness checks)
so the 2x2 (support x truth) analyses can be run and re-run offline with any
support split (median, terciles) and any matching scheme.

SUPPORT is estimated per fact as the mean LOPO agreement: for each paraphrase,
the agreement of the OTHER five paraphrases' answers (how consistently the model
gives the same answer under DIFFERENT wording), then averaged. This is
independent of the target response's own hidden state, which is what breaks the
circularity of estimating stability from the probed response itself.

We extract ONE hidden-state vector per fact (greedy answer, last token), at
several depth fractions, exactly as the other extractors, so results are
comparable to the main analysis.

Saves outputs/hidden2x2_<model>.npz with:
    X_<frac>  float16 [n, hidden]
    support   float32 [n]     LOPO agreement in [0,1]
    correct   int8    [n]     1=correct, 0=incorrect
    entropy   float32 [n]     answer entropy (nats)
    fracs     float   [k]
Usage:
    python extract_hidden_2x2.py --model qwen3-8b --n 6000
    python extract_hidden_2x2.py --model all --n 6000
"""
from __future__ import annotations
import argparse, os, re
from collections import Counter
import numpy as np
import env_load  # noqa
from schema import load_dataset
from registry import get_model, MODELS
from distributed import load_distributed


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def lopo_support_and_truth(by_paraphrase):
    """Return (mean_support, correct_bool, entropy) or None.
    support: mean over paraphrases of the OTHER-paraphrases' modal agreement.
    correct: majority correctness over all samples.
    entropy: Shannon entropy over all normalized answers."""
    paras = []
    all_answers, all_correct = [], []
    for pa in by_paraphrase:
        ans = [_norm(a) for a in pa.get("answers", [])]
        cor = pa.get("correct", [])
        paras.append(ans)
        all_answers.extend(ans)
        all_correct.extend(cor)
    n_para = len(paras)
    if n_para < 2 or not all_correct:
        return None
    sups = []
    for i in range(n_para):
        others = []
        for j in range(n_para):
            if j != i:
                others.extend(paras[j])
        if not others:
            continue
        modal = Counter(others).most_common(1)[0][1]
        sups.append(modal / len(others))
    if not sups:
        return None
    support = float(np.mean(sups))
    correct = sum(all_correct) / len(all_correct) >= 0.5
    # entropy
    import math
    c = Counter(all_answers); tot = len(all_answers)
    ent = -sum((v / tot) * math.log(v / tot) for v in c.values()) if tot else 0.0
    return support, bool(correct), float(ent)


def collect(model, tok, question, device, fracs, max_new=24):
    import torch
    prompt = question
    if getattr(tok, "chat_template", None):
        for kw in ({"enable_thinking": False}, {}):
            try:
                prompt = tok.apply_chat_template(
                    [{"role": "user", "content": question}],
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


def run(model_key, n=6000, fracs=(0.4, 0.6, 0.8, 1.0)):
    spec = get_model(model_key)
    path = f"outputs/probed_{model_key}.jsonl"
    if not os.path.exists(path):
        print(f"[skip] {model_key}: no probed set"); return
    data = load_dataset(path)
    items = []
    for x in data:
        if x.item_type != "fact":
            continue
        stab = getattr(x, "stability", None) or {}
        bp = stab.get("by_paraphrase")
        if not bp:
            continue
        st = lopo_support_and_truth(bp)
        if st is None:
            continue
        items.append((x, st[0], st[1], st[2]))
    items = items[:n]
    if not items:
        print(f"[skip] {model_key}: no facts with by_paraphrase"); return
    n_correct = sum(1 for _, _, c, _ in items if c)
    print(f"[2x2] {model_key}: {len(items)} facts "
          f"(correct={n_correct}, incorrect={len(items)-n_correct})")

    model, tok = load_distributed(spec, for_training=False)
    model.eval()
    dev = next(model.parameters()).device

    buckets = {f: [] for f in fracs}
    supports, corrects, ents = [], [], []
    for i, (inst, sup, cor, ent) in enumerate(items):
        try:
            h = collect(model, tok, inst.question, dev, fracs)
        except Exception as e:
            print(f"  (skip {inst.id}: {e})"); continue
        for f in fracs:
            buckets[f].append(h[f])
        supports.append(sup); corrects.append(1 if cor else 0); ents.append(ent)
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(items)}")

    out = {f"X_{f}": np.array(buckets[f], dtype=np.float16) for f in fracs}
    out["support"] = np.array(supports, dtype=np.float32)
    out["correct"] = np.array(corrects, dtype=np.int8)
    out["entropy"] = np.array(ents, dtype=np.float32)
    out["fracs"] = np.array(fracs, dtype=float)
    os.makedirs("outputs", exist_ok=True)
    fn = f"outputs/hidden2x2_{model_key}.npz"
    np.savez_compressed(fn, **out)
    print(f"[2x2] saved {fn} (n={len(supports)})")
    del model
    import torch, gc; gc.collect(); torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3-8b")
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--layers", default="0.4,0.6,0.8,1.0")
    args = p.parse_args()
    fracs = tuple(float(x) for x in args.layers.split(","))
    keys = [m.key for m in MODELS if m.access == "weights"] \
        if args.model == "all" else [args.model]
    for k in keys:
        run(k, n=args.n, fracs=fracs)


if __name__ == "__main__":
    main()
