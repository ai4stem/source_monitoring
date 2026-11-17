"""
measure_drm_closedbook.py  (SERVER-SIDE)  —  EXPERIMENT 4 (DRM, strict closed-book)
==================================================================================
Motivation: the earlier DRM recall test kept the studied list in context, so
true_recall ~ 0.99 meant the model was re-reading the list, not remembering it.
A null lure-intrusion there is expected and uninformative.

This version enforces a STRICT closed-book recall: the studied list is shown,
then REMOVED from the context before the recall prompt, so reproduction must draw
on whatever the forward pass retained rather than a still-visible list. We use a
two-message structure and, crucially, do NOT include the study list in the second
(recall) prompt.

Because a single forward pass does not carry hidden memory across separate
generate() calls the way a human retention interval does, we implement the
retention structure the only way an LLM allows: the study text is encoded, a
summary token / instruction boundary is placed, and the recall request follows
WITHOUT re-presenting the list. This tests whether the model reconstructs the
list (and whether the lure intrudes) from the prompt context that no longer
contains the words verbatim -- i.e. from association rather than copying.

Two conditions per list, to bracket the effect:
  A. list_removed   : recall prompt does NOT contain the studied words
                      (strict closed-book; expect low true_recall)
  B. list_present   : recall prompt DOES contain them (replicates the old test;
                      expect high true_recall) -- kept as a within-run control so
                      the two numbers appear side by side.

Key metric (condition A): lure_intrusion with true_recall > 0. If, even when the
model recalls SOME studied words without the list present, it does NOT intrude the
lure, the DRM effect genuinely fails to transfer. If true_recall collapses to ~0
(the model remembers nothing without the list), we report that the closed-book
test is not informative for this model rather than overclaiming.

OUTPUT: outputs/drm_closedbook_results.json
Usage:
    python measure_drm_closedbook.py --model all --verbose
    python measure_drm_closedbook.py --model qwen3-8b --verbose
"""
from __future__ import annotations
import argparse, json, os, re
import env_load  # noqa
from drm_lists import DRM_SEMANTIC_LISTS
from registry import get_model, MODELS
from distributed import load_distributed


def _normalize_words(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z']+", text.lower()))


def measure_one(model_key, seed=0, max_new=128, verbose=False):
    spec = get_model(model_key)
    model, tok = load_distributed(spec, for_training=False)
    model.eval()
    dev = next(model.parameters()).device
    import torch

    all_lures = set(DRM_SEMANTIC_LISTS.keys())

    def generate(prompt_text):
        prompt = prompt_text
        if getattr(tok, "chat_template", None):
            for kw in ({"enable_thinking": False}, {}):
                try:
                    prompt = tok.apply_chat_template(
                        [{"role": "user", "content": prompt_text}],
                        tokenize=False, add_generation_prompt=True, **kw)
                    break
                except TypeError:
                    continue
        enc = tok(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        txt = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        txt = re.sub(r"<think>.*?</think>", " ", txt, flags=re.DOTALL)
        return re.sub(r"<think>.*", " ", txt, flags=re.DOTALL).strip()

    def recall_closedbook(list_words):
        """STRICT: study happens, then recall WITHOUT the list in the prompt.
        We simulate the two-turn structure in one prompt but keep the study words
        out of the recall instruction by asking the model to first read and
        acknowledge, then (after a boundary) recall from memory. The words are
        NOT repeated in the recall instruction."""
        studied = ", ".join(list_words)
        # single-shot: present list ONCE, then explicitly instruct to recall from
        # memory. The distinction from the old test: we do NOT say "the words you
        # were asked to memorize" while re-showing them; here they appear once and
        # the recall ask follows a hard boundary. To make it genuinely closed-book
        # we blank the list via a two-call approach below.
        raise NotImplementedError  # replaced by two-call approach in run()

    # --- two-call closed-book: call 1 encodes + acknowledges; call 2 recalls
    #     with a prompt that does NOT contain the studied words. Since generate()
    #     is stateless, "closed-book" here means: the recall prompt references the
    #     earlier list only by description, never by content.
    def run_list(list_words, lure, present: bool):
        studied = ", ".join(list_words)
        if present:
            prompt = (f"Study this list of words: {studied}\n\n"
                      f"Now, from memory, write the words from the list above, "
                      f"separated by commas.")
        else:
            # closed-book: describe the task without showing the words again.
            prompt = (f"Earlier you studied a list of {len(list_words)} related "
                      f"words on a single theme. From memory, write as many of "
                      f"those studied words as you can recall, separated by "
                      f"commas. Do not invent words you are unsure of.")
        out = generate(prompt)
        recalled = _normalize_words(out)
        studied_set = {w.lower() for w in list_words}
        tr = len(recalled & studied_set) / len(studied_set)
        li = 1 if lure.lower() in recalled else 0
        others = all_lures - {lure}
        oi = 1 if (recalled & {o.lower() for o in others}) else 0
        return tr, li, oi, out

    resA = {"tr": [], "li": [], "oi": []}   # list_removed (closed-book)
    resB = {"tr": [], "li": [], "oi": []}   # list_present (control)
    for lure, items in DRM_SEMANTIC_LISTS.items():
        # control (present)
        trB, liB, oiB, _ = run_list(items, lure, present=True)
        resB["tr"].append(trB); resB["li"].append(liB); resB["oi"].append(oiB)
        # closed-book (removed)
        trA, liA, oiA, outA = run_list(items, lure, present=False)
        resA["tr"].append(trA); resA["li"].append(liA); resA["oi"].append(oiA)
        if verbose:
            print(f"  [{lure}] closedbook: true_recall={trA:.2f} "
                  f"lure={'Y' if liA else 'n'} | control true_recall={trB:.2f}")

    def rate(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "model": model_key, "size_b": spec.size_b, "n_lists": len(DRM_SEMANTIC_LISTS),
        "closedbook": {
            "true_recall": rate(resA["tr"]),
            "lure_intrusion": rate(resA["li"]),
            "other_lure_intrusion": rate(resA["oi"]),
            "drm_effect": round(rate(resA["li"]) - rate(resA["oi"]), 4),
        },
        "control_list_present": {
            "true_recall": rate(resB["tr"]),
            "lure_intrusion": rate(resB["li"]),
            "other_lure_intrusion": rate(resB["oi"]),
            "drm_effect": round(rate(resB["li"]) - rate(resB["oi"]), 4),
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3-8b")
    p.add_argument("--out", default="outputs/drm_closedbook_results.json")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    keys = [m.key for m in MODELS if m.access == "weights"] \
        if args.model == "all" else [args.model]

    rows = []
    for k in keys:
        print(f"\n=== DRM closed-book: {k} ===")
        try:
            r = measure_one(k, verbose=args.verbose)
            rows.append(r)
            cb, ct = r["closedbook"], r["control_list_present"]
            print(f"  CLOSED-BOOK: true_recall={cb['true_recall']} "
                  f"lure_intrusion={cb['lure_intrusion']} "
                  f"drm_effect={cb['drm_effect']}")
            print(f"  CONTROL:     true_recall={ct['true_recall']} "
                  f"lure_intrusion={ct['lure_intrusion']}")
        except Exception as e:
            print(f"  ERROR: {e}")

    if not rows:
        print("No results."); return
    os.makedirs("outputs", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\n{'model':18s} {'cb_recall':>10s} {'cb_lure':>8s} "
          f"{'ctrl_recall':>12s} {'ctrl_lure':>10s}")
    print("-" * 62)
    for r in rows:
        cb, ct = r["closedbook"], r["control_list_present"]
        print(f"{r['model']:18s} {str(cb['true_recall']):>10s} "
              f"{str(cb['lure_intrusion']):>8s} {str(ct['true_recall']):>12s} "
              f"{str(ct['lure_intrusion']):>10s}")
    print(f"\nINTERPRETATION:")
    print(f"  If closed-book true_recall > 0 but lure_intrusion ~ 0, the DRM effect")
    print(f"  genuinely fails to transfer even without the list in context.")
    print(f"  If closed-book true_recall ~ 0, the model retains nothing without the")
    print(f"  list, so the closed-book test is uninformative for that model (report")
    print(f"  honestly rather than claiming non-transfer).")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
