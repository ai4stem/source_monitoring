"""
measure_conflict.py  (SERVER-SIDE)  —  EXPERIMENT 4
===================================================
Bring the knowledge-conflict data into the Results with a REAL generation pass.

Motivation: the context-mass asymmetry (verbatim attends to context,
~0.78) is close to true by construction. Knowledge conflict is the one condition
that can reveal genuine source misattribution: the context asserts X while the
model's parametric knowledge says Y. What the model does then, and whether its
internal signals track it, is a real finding rather than a manipulation check.

For each conflict item (context contradicts parametric answer) and each
non-conflict control (context agrees / is irrelevant), we generate the answer and
score:

  followed_context  : answer matches the context-asserted target (source=verbatim)
  followed_param    : answer matches the model's parametric answer instead
  verbatim_override : among conflict items, fraction where the model followed the
                      context over its own knowledge
  over_deference    : among NON-conflict controls, fraction where the model
                      deferred to an irrelevant/agreeing context instead of
                      answering from knowledge (a false-context-following baseline)

This replaces the proxy in run_comparison.py (which derived behaviour from the
probe prediction) with an actual generate() call.

INPUT : eval_<model>.jsonl  (must contain probe_kind in {conflict, non_conflict_control})
OUTPUT: outputs/conflict_<model>.json  and appends to outputs/conflict_all.json

Usage:
    python measure_conflict.py --model qwen3-8b --verbose
    python measure_conflict.py --model all
"""
from __future__ import annotations
import argparse, json, os, glob, re
import env_load  # noqa
from schema import load_dataset
from registry import get_model, MODELS
from distributed import load_distributed


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _contains(needle, hay):
    n = _norm(str(needle))
    return bool(n) and n in _norm(hay)


def measure(model_key, verbose=False, max_new=48):
    spec = get_model(model_key)
    path = f"outputs/eval_{model_key}.jsonl"
    if not os.path.exists(path):
        print(f"[skip] {model_key}: no eval set"); return None
    data = load_dataset(path)
    conflict = [x for x in data if x.probe_kind == "conflict"]
    control = [x for x in data if x.probe_kind == "non_conflict_control"]
    if not conflict and not control:
        print(f"[skip] {model_key}: no conflict/control items in eval set")
        return None

    import torch
    model, tok = load_distributed(spec, for_training=False)
    model.eval()
    dev = next(model.parameters()).device

    def answer(inst):
        base = (inst.context + "\n\n") if inst.context else ""
        user = base + "Q: " + inst.question + "\nA:"
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
        return re.sub(r"<think>.*?</think>", " ", txt, flags=re.DOTALL).strip()

    # conflict: target = context-asserted answer; parametric_lure = model's own answer
    foll_ctx, foll_par = [], []
    for inst in conflict:
        out = answer(inst)
        ctx_target = inst.target
        par = inst.parametric_lure
        fc = _contains(ctx_target, out)
        fp = _contains(par, out) if par else False
        foll_ctx.append(1 if fc else 0)
        foll_par.append(1 if fp else 0)
        if verbose:
            print(f"  [conflict] ctx='{ctx_target}' par='{par}' -> "
                  f"{'CTX' if fc else ('PAR' if fp else '??')} : {out[:50]}")

    # non-conflict control: context is irrelevant/agreeing; target is the true answer
    over_def = []
    for inst in control:
        out = answer(inst)
        # over-deference proxy: model echoes a context detail that is NOT the answer.
        # Here we score whether it still gets the correct target (good) vs not.
        correct = _contains(inst.target, out)
        over_def.append(0 if correct else 1)

    def rate(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    res = {
        "model": model_key, "size_b": spec.size_b,
        "n_conflict": len(conflict), "n_control": len(control),
        "verbatim_override": rate(foll_ctx),      # followed context over knowledge
        "followed_parametric": rate(foll_par),    # kept own answer under conflict
        "over_deference": rate(over_def),         # failed control (deferred wrongly)
    }
    del model
    import gc; gc.collect(); torch.cuda.empty_cache()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3-8b")
    p.add_argument("--out", default="outputs/conflict_all.json")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    keys = [m.key for m in MODELS if m.access == "weights"] \
        if args.model == "all" else [args.model]

    rows = []
    for k in keys:
        print(f"[conflict] {k}")
        r = measure(k, verbose=args.verbose)
        if r:
            rows.append(r)
            with open(f"outputs/conflict_{k}.json", "w") as f:
                json.dump(r, f, indent=2)
            print(f"   verbatim_override={r['verbatim_override']} "
                  f"followed_parametric={r['followed_parametric']} "
                  f"over_deference={r['over_deference']} "
                  f"(n_conflict={r['n_conflict']})")
    if not rows:
        print("\nNo results. Check that eval_<model>.jsonl contains conflict items.")
        return
    rows.sort(key=lambda r: r["size_b"])
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)

    import statistics as st
    vo = [r["verbatim_override"] for r in rows if r["verbatim_override"] is not None]
    print(f"\n{'model':18s} {'size':>5s} {'vrb_override':>12s} "
          f"{'foll_param':>11s} {'over_defer':>11s}")
    print("-" * 60)
    for r in rows:
        print(f"{r['model']:18s} {r['size_b']:>5.1f} "
              f"{str(r['verbatim_override']):>12s} {str(r['followed_parametric']):>11s} "
              f"{str(r['over_deference']):>11s}")
    if vo:
        print("-" * 60)
        print(f"{'MEAN override':18s} {'':>5s} {st.mean(vo):>12.3f}")
    print(f"\nThis is the real source-misattribution condition: under conflict, does "
          f"the model follow the context (verbatim) or its own parametric answer? "
          f"Reporting this turns the reality-monitoring section from a near-tautology "
          f"into an actual behavioural finding.")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
