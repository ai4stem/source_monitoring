"""
diagnose_probing.py  (SERVER-SIDE diagnostic)
=============================================
Prints what the model ACTUALLY generates for a handful of easy facts during
probing, so we can see why parametric labels aren't appearing. Run this on the
server with the model that showed parametric_available=1.

Usage:
    python diagnose_probing.py --model qwen3-8b --n 10
"""
from __future__ import annotations
import env_load  # noqa
from load_external import load_triviaqa_facts
from probe_knowledge import _is_correct, _normalize, stability_record
from distributed import load_distributed


def run(model_key="qwen3-8b", n=10, use_chat_template=True, source="squad"):
    from registry import get_model
    import torch
    spec = get_model(model_key)
    model, tok = load_distributed(spec, for_training=False)
    model.eval()
    device = next(model.parameters()).device

    if source == "squad":
        from load_external import load_squad_facts
        facts = load_squad_facts(n, seed=0)
    elif source == "popqa":
        from load_external import load_popqa_facts
        facts = load_popqa_facts(n, seed=0)
    else:
        facts = load_triviaqa_facts(n, seed=0)
    print(f"\n=== Probing diagnostic for {model_key} "
          f"(source={source}, chat_template={use_chat_template}) ===\n")

    for inst in facts:
        q = inst.question
        gold = inst.answer_aliases or [inst.target]

        # build prompt two ways
        if use_chat_template and hasattr(tok, "apply_chat_template"):
            msgs = [{"role": "user", "content": q}]
            prompt = None
            for kwargs in ({"enable_thinking": False}, {}):
                try:
                    prompt = tok.apply_chat_template(msgs, tokenize=False,
                                                     add_generation_prompt=True, **kwargs)
                    break
                except TypeError:
                    continue
            if prompt is None:
                prompt = q
        else:
            prompt = q

        enc = tok(prompt, return_tensors="pt").to(device)
        if inst is facts[0]:
            print(f"--- formatted prompt (first item) ---\n{prompt}\n"
                  f"--- ({enc['input_ids'].shape[1]} tokens) ---\n")
        outs = []
        for _ in range(3):
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=128, do_sample=True,
                                     temperature=0.7,
                                     pad_token_id=tok.eos_token_id)
            raw = tok.decode(gen[0][enc["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip()
            n_gen = gen.shape[1] - enc["input_ids"].shape[1]
            has_think = "<think>" in raw
            import re
            txt = re.sub(r"<think>.*?</think>", " ", raw, flags=re.DOTALL)
            txt = re.sub(r"<think>.*", " ", txt, flags=re.DOTALL).strip()
            outs.append((txt, n_gen, has_think))

        hits = [_is_correct(gold, o[0]) for o in outs]
        n_hit = sum(hits)
        verdict = "PARAMETRIC" if n_hit >= 2 else ("confab" if n_hit == 0 else "gray")
        print(f"Q: {q}")
        print(f"   gold: {gold[:3]}{'...' if len(gold) > 3 else ''}  -> {n_hit}/3 hit [{verdict}]")
        for (o, n_gen, has_think), h in zip(outs, hits):
            think_flag = " THINK!" if has_think else ""
            print(f"   {'HIT ' if h else 'miss'} [{n_gen}tok{think_flag}] -> {o[:180]!r}")
        print()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3-8b")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--source", default="squad", choices=["squad", "triviaqa", "popqa"])
    p.add_argument("--no_chat_template", action="store_true")
    args = p.parse_args()
    run(args.model, args.n, use_chat_template=not args.no_chat_template,
        source=args.source)
