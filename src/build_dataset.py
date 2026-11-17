"""
build_dataset.py
================
Constructs the source-monitoring dataset from:
  - the 36 normed DRM word lists (McDermott & Watson 2001 semantic associates,
    transcribed verbatim in drm_lists.py)
  - factual QA seeds (parametric vs confabulated)
  - knowledge-conflict templates

Each DRM list expands into the 3 control probes we designed:
    critical_lure (target=no, source=confabulated if model errs)
    studied       (target=yes, source=verbatim)        [positive control]
    unrelated     (target=no,  no label)               [negative control]

NOTE on parametric/confabulated labels: the *final* source label for fact items
is model-relative and is assigned later by stability probing (probe_knowledge.py).
Here fact items carry source='pending' until probing resolves them.
"""

from __future__ import annotations
from schema import Instance, Claim, save_dataset, dataset_summary
from drm_lists import DRM_SEMANTIC_LISTS, SOURCE as DRM_SOURCE
import json
import random


# ---------------------------------------------------------------------------
# 1. DRM lists: build from the 36 normed lists. Words are in BAS-descending
#    order; we bucket each list's overall strength qualitatively by its size and
#    keep the ordering intact for position-based analyses.
#    (For per-list false-recall norms you can later attach Roediger et al. 2001
#     Appendix B values; here every normed list is included.)
# ---------------------------------------------------------------------------
def _make_drm_lists() -> list[dict]:
    out = []
    for lure, items in DRM_SEMANTIC_LISTS.items():
        out.append({"lure": lure, "bas": "normed", "items": list(items)})
    return out

DRM_LISTS = _make_drm_lists()


# Unrelated lures (for negative-control probes) -- semantically far from any list.
def build_drm_items(lists=DRM_LISTS, seed: int = 0) -> list[Instance]:
    rng = random.Random(seed)
    out: list[Instance] = []
    for li, entry in enumerate(lists):
        lure, items, bas = entry["lure"], entry["items"], entry["bas"]
        list_text = ", ".join(items)
        context = f"Study this list of words carefully: {list_text}."

        # (a) critical lure probe  -> confabulated if model says "yes"
        out.append(Instance(
            id=f"drm_{lure}_lure", item_type="drm", probe_kind="critical_lure",
            context=context, question=f"Was the word '{lure}' in the list?",
            target="No, that word was not in the list.",
            claims=[Claim(text=f"'{lure}' was in the list",
                          source="confabulated", conflict=0, labeled=True)],
            lure=lure, list_type=f"{bas}_BAS",
        ))

        # (b) studied-item probe  -> verbatim (positive control)
        studied = rng.choice(items)
        out.append(Instance(
            id=f"drm_{lure}_studied", item_type="drm", probe_kind="studied",
            context=context, question=f"Was the word '{studied}' in the list?",
            target="Yes, that word was in the list.",
            claims=[Claim(text=f"'{studied}' was in the list",
                          source="verbatim", conflict=0, labeled=True)],
            lure=lure, list_type=f"{bas}_BAS",
        ))

        # (c) unrelated-lure probe -> no label (negative control)
        # Pick a real word from a DIFFERENT list that does NOT appear in this
        # list and is not this list's lure -- a genuine word, guaranteed absent,
        # semantically unrelated to the studied set.
        unrelated = _pick_unrelated(lure, items, rng)
        out.append(Instance(
            id=f"drm_{lure}_unrelated", item_type="drm", probe_kind="unrelated",
            context=context, question=f"Was the word '{unrelated}' in the list?",
            target="No, that word was not in the list.",
            claims=[Claim(text=f"'{unrelated}' was in the list",
                          source=None, conflict=0, labeled=False)],
            lure=lure, list_type=f"{bas}_BAS",
        ))
    return out


def _pick_unrelated(lure: str, items: list[str], rng) -> str:
    """A real word from another DRM list, absent from this list and not its lure."""
    present = {w.lower() for w in items} | {lure.lower()}
    # candidate lures from OTHER lists that do NOT appear anywhere in this list
    candidates = [l for l in DRM_SEMANTIC_LISTS if l.lower() not in present]
    if not candidates:
        candidates = ["umbrella"]
    return rng.choice(candidates)


# ---------------------------------------------------------------------------
# 2. Fact items.  Provisional kind only; stability probing assigns final label.
#    `kind`: "known_seed"  -> expected parametric (verify via probing)
#            "fictional"   -> guaranteed confabulated if answered confidently
# ---------------------------------------------------------------------------
KNOWN_FACTS = [
    ("What is the capital of France?", "Paris"),
    ("What is the chemical symbol for gold?", "Au"),
    ("Who wrote the play 'Hamlet'?", "William Shakespeare"),
    ("What is the largest planet in the solar system?", "Jupiter"),
    ("What is the speed of light in vacuum (approx, m/s)?", "299792458"),
]
FICTIONAL_FACTS = [
    "In what year did the chemist Almira Tetractys publish her treatise on phlogiston?",
    "Which river flows through the ancient city of Quellanor?",
    "Who discovered the element veridium?",
    "What is the population of the town of Brassford-on-Wye in 1850?",
]


def build_fact_items() -> list[Instance]:
    out: list[Instance] = []
    for qi, (q, a) in enumerate(KNOWN_FACTS):
        out.append(Instance(
            id=f"fact_known_{qi}", item_type="fact", probe_kind="parametric_fact",
            context="", question=q, target=a,
            # source 'pending': probing assigns parametric / gray-zone(labeled=False)
            claims=[Claim(text=a, source="pending", conflict=0, labeled=True)],
        ))
    for qi, q in enumerate(FICTIONAL_FACTS):
        out.append(Instance(
            id=f"fact_fictional_{qi}", item_type="fact", probe_kind="confabulated_fact",
            context="", question=q, target="[unanswerable: entity does not exist]",
            claims=[Claim(text="<model-generated answer>",
                          source="confabulated", conflict=0, labeled=True)],
        ))
    return out


# ---------------------------------------------------------------------------
# 3. Conflict items.  context contradicts (or is neutral to) parametric knowledge.
# ---------------------------------------------------------------------------
CONFLICT_TEMPLATES = [
    # (context, question, verbatim_answer, parametric_lure or None)
    ("In this alternate scenario, the capital of Australia is Sydney, where the "
     "federal parliament convenes.",
     "According to the passage, what is the capital of Australia?",
     "Sydney", "Canberra"),
    ("In the nation of Veldara, the capital city is Morrowind. All government "
     "offices are located in Morrowind.",
     "According to the passage, what is the capital of Veldara?",
     "Morrowind", None),  # fictional country -> pure verbatim, no parametric lure
    ("For the purposes of this story, water boils at 150 degrees Celsius at sea level.",
     "According to the passage, at what temperature does water boil at sea level?",
     "150 degrees Celsius", "100 degrees Celsius"),
]
# non-conflict control: context agrees with / is irrelevant to parametric truth
NON_CONFLICT_TEMPLATES = [
    ("The capital of Japan is Tokyo, a major global financial center.",
     "What is the capital of Japan?", "Tokyo"),
    ("Photosynthesis converts light energy into chemical energy in plants.",
     "What process converts light energy into chemical energy in plants?", "Photosynthesis"),
]


def build_conflict_items() -> list[Instance]:
    out: list[Instance] = []
    for ci, (ctx, q, ans, lure) in enumerate(CONFLICT_TEMPLATES):
        out.append(Instance(
            id=f"conflict_{ci}", item_type="conflict", probe_kind="conflict",
            context=ctx, question=q, target=ans,
            parametric_lure=lure,
            claims=[Claim(text=ans, source="verbatim", conflict=1, labeled=True)],
            context_char_span=[0, len(ctx)],
        ))
    for ci, (ctx, q, ans) in enumerate(NON_CONFLICT_TEMPLATES):
        out.append(Instance(
            id=f"nonconflict_{ci}", item_type="conflict", probe_kind="non_conflict_control",
            context=ctx, question=q, target=ans,
            # answer is parametrically true; tests over-deference
            claims=[Claim(text=ans, source="parametric", conflict=0, labeled=True)],
            context_char_span=[0, len(ctx)],
        ))
    return out


def build_all(seed: int = 0) -> list[Instance]:
    items = build_drm_items(seed=seed) + build_fact_items() + build_conflict_items()
    return items


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/dataset.jsonl")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    items = build_all(seed=args.seed)
    save_dataset(items, args.out)
    summary = dataset_summary(items)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved {len(items)} instances -> {args.out}")
