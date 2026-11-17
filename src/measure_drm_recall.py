"""
measure_drm_recall.py  (SERVER-SIDE)
====================================
DRM false memory measured the RIGHT way for LLMs: recall with a distractor.

Recognition ("was 'bad' in the list?") fails on LLMs because the list sits in
context -- the model just searches the text and says no. Human DRM instead uses
RECALL after a delay: study a list, do something else, then recall. The critical
lure intrudes into recall because memory is stored as gist.

We emulate the study-delay-recall structure:
  1. STUDY:    show the list, ask the model to memorize it
  2. DISTRACT: insert a long unrelated passage (pushes the list out of the
               attention focus, emulating a retention interval)
  3. RECALL:   ask the model to list the words it memorized
  4. SCORE:    does the (non-presented) critical lure INTRUDE into the recall?

Measures per list:
  lure_intrusion  : critical lure appears in recall (NOT in the studied list)
  true_recall     : fraction of studied words recalled (positive control;
                    if ~0, the distractor destroyed memory -> measurement invalid)
  other_lure_intr : OTHER lists' lures appearing (baseline gist/false-alarm)

A DRM effect = lure_intrusion >> other_lure_intr, with true_recall well above 0
(the model does remember the list, yet still intrudes the lure).

Usage:
  python measure_drm_recall.py --model gemma3-12b --distractor 400 --verbose
  python measure_drm_recall.py --model all --distractor 400
"""
from __future__ import annotations
import argparse, json, os, re
import env_load  # noqa
from drm_lists import DRM_SEMANTIC_LISTS
from registry import get_model, list_keys
from distributed import load_distributed


# A neutral distractor passage, repeated/truncated to a target word count. The
# content is deliberately unrelated to any DRM list (geography/process text).
_DISTRACTOR_SRC = (
    "The river system in the northern basin flows through a series of shallow "
    "valleys before reaching the coastal plain. Surveyors mapped the region over "
    "several seasons, recording elevation, soil composition, and seasonal water "
    "levels. Transport along the main channel depends on the depth of the "
    "central passage, which shifts gradually as sediment accumulates near the "
    "delta. Engineers scheduled maintenance of the locks during the dry months, "
    "when traffic is lightest and repairs can proceed without interrupting the "
    "flow of goods. Records from the regional office describe how the network of "
    "canals expanded over decades, connecting inland markets to the harbor and "
    "supporting the steady growth of trade in timber, grain, and stone."
)


def _make_distractor(n_words: int) -> str:
    words = _DISTRACTOR_SRC.split()
    out = []
    while len(out) < n_words:
        out.extend(words)
    return " ".join(out[:n_words])


def _normalize_words(text: str) -> set[str]:
    toks = re.findall(r"[a-zA-Z']+", text.lower())
    return set(toks)


def measure_one(model_key, distractor_words=400, seed=0, max_new=128,
                verbose=False):
    spec = get_model(model_key)
    if spec.access != "weights":
        raise NotImplementedError(f"{model_key} is not a weights model")
    model, tok = load_distributed(spec, for_training=False)
    model.eval()
    dev = next(model.parameters()).device
    import torch

    distractor = _make_distractor(distractor_words)
    all_lures = set(DRM_SEMANTIC_LISTS.keys())

    def recall(list_words):
        studied = ", ".join(list_words)
        user = (f"Memorize this list of words:\n{studied}\n\n"
                f"Now read the following unrelated passage:\n{distractor}\n\n"
                f"From memory, list as many of the words you were asked to "
                f"memorize as you can. Give only the words, separated by commas.")
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
        txt = re.sub(r"<think>.*?</think>", " ", txt, flags=re.DOTALL)
        txt = re.sub(r"<think>.*", " ", txt, flags=re.DOTALL).strip()
        return txt

    lure_intrusions, true_recalls, other_intrusions = [], [], []
    for lure, items in DRM_SEMANTIC_LISTS.items():
        out = recall(items)
        recalled = _normalize_words(out)
        studied_set = {w.lower() for w in items}

        # true recall: fraction of studied words that appear
        tr = len(recalled & studied_set) / len(studied_set)
        # critical lure intrusion: the lure appears though it was NOT studied
        li = 1 if lure.lower() in recalled else 0
        # other-list lure intrusion: any OTHER list's lure appears
        others = all_lures - {lure}
        oi = 1 if (recalled & {o.lower() for o in others}) else 0

        true_recalls.append(tr)
        lure_intrusions.append(li)
        other_intrusions.append(oi)
        if verbose:
            print(f"  [{lure}] true_recall={tr:.2f} lure_intruded={'Y' if li else 'n'} "
                  f"recall='{out[:70]}'")

    def rate(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "model": model_key, "size_b": spec.size_b,
        "distractor_words": distractor_words,
        "lure_intrusion": rate(lure_intrusions),      # DRM false memory
        "true_recall": round(sum(true_recalls) / len(true_recalls), 4),
        "other_lure_intrusion": rate(other_intrusions),  # baseline
        "drm_effect": round(rate(lure_intrusions) - rate(other_intrusions), 4),
        "n_lists": len(DRM_SEMANTIC_LISTS),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma3-12b")
    p.add_argument("--distractor", type=int, default=400,
                   help="distractor length in words (retention interval)")
    p.add_argument("--out", default="outputs/drm_recall_results.json")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    keys = [k for k in list_keys() if get_model(k).access == "weights"] \
        if args.model == "all" else [args.model]

    results = []
    for k in keys:
        print(f"\n=== DRM recall: {k} (distractor={args.distractor} words) ===")
        try:
            r = measure_one(k, distractor_words=args.distractor, verbose=args.verbose)
            results.append(r)
            print(f"  lure_intrusion={r['lure_intrusion']} "
                  f"true_recall={r['true_recall']} "
                  f"other_intrusion={r['other_lure_intrusion']} "
                  f"drm_effect={r['drm_effect']}")
        except Exception as e:
            print(f"  ERROR: {e}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {args.out}")
    print(f"\n{'model':18s} {'size':>5s} {'lure_intr':>10s} "
          f"{'true_recall':>12s} {'other_intr':>11s} {'drm_effect':>11s}")
    print("-" * 72)
    for r in results:
        print(f"{r['model']:18s} {r['size_b']:>5.1f} "
              f"{str(r['lure_intrusion']):>10s} {str(r['true_recall']):>12s} "
              f"{str(r['other_lure_intrusion']):>11s} {str(r['drm_effect']):>11s}")


if __name__ == "__main__":
    main()
