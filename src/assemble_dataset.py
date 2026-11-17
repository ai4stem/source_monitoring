"""
assemble_dataset.py  (SERVER-SIDE: needs model for probing)
===========================================================
Turns the oversampled external pools + DRM controls into the final training set,
using model-relative stability probing to assign clean parametric/confabulated
labels and BALANCING to target per-class counts.

Flow ("oversample then filter", as agreed):
  1. build DRM controls (verbatim/confabulated, no probing needed)   -> drm
  2. download oversampled facts (TriviaQA + PopQA) + conflict (ConflictQA)
  3. PROBE every fact candidate on THIS model -> parametric / confab / gray-zone
  4. keep gray-zone OUT; sample to target counts, prioritizing the scarce class
     (confabulated), so classes are balanced enough for macro-F1
  5. hold out an OOD pool (NQ) untouched for generalization eval

Targets are the "sweet spot" we discussed (~5-6k train), tunable via args.
A data-scaling curve can be produced by re-running with smaller `target_*`.

Only runnable where a model + `datasets` are available; the balancing/sampling
logic below (assemble_from_labeled) is pure-python and unit-testable.
"""

from __future__ import annotations
from schema import Instance, Claim, save_dataset, dataset_summary
from collections import Counter
import random


# ---------------------------------------------------------------------------
# Pure-python balancing (testable without a model)
# ---------------------------------------------------------------------------
def assemble_from_labeled(drm_items, fact_items, conflict_items,
                          target_parametric=2500, target_confab=800,
                          target_conflict=1000, seed=0) -> list[Instance]:
    """Given already-labeled items, sample to target counts.

    drm_items:      DRM controls (already verbatim/confabulated/unlabeled)
    fact_items:     facts whose claim.source is resolved (parametric/confabulated)
                    or 'pending'->dropped, or labeled=False (gray-zone)->dropped
    conflict_items: conflict items (verbatim, kappa=1)
    """
    rng = random.Random(seed)

    # split resolved facts by source
    param = [i for i in fact_items
             if i.claims and i.claims[0].is_resolved()
             and i.claims[0].source == "parametric"]
    confab = [i for i in fact_items
              if i.claims and i.claims[0].is_resolved()
              and i.claims[0].source == "confabulated"]
    rng.shuffle(param); rng.shuffle(confab)
    rng.shuffle(conflict_items)

    chosen_param = param[:target_parametric]
    chosen_confab = confab[:target_confab]
    chosen_conflict = conflict_items[:target_conflict]

    final = list(drm_items) + chosen_param + chosen_confab + chosen_conflict
    rng.shuffle(final)
    return final, {
        "drm": len(drm_items),
        "parametric_available": len(param), "parametric_used": len(chosen_param),
        "confabulated_available": len(confab), "confabulated_used": len(chosen_confab),
        "conflict_available": len(conflict_items), "conflict_used": len(chosen_conflict),
    }


# ---------------------------------------------------------------------------
# Full server-side assembly (needs model + datasets)
# ---------------------------------------------------------------------------
def assemble_full(model, tokenizer, device="cuda",
                  n_squad=0, n_trivia=8000, n_popqa=3000, n_conflict=1500,
                  target_parametric=2500, target_confab=800, target_conflict=1000,
                  n_paraphrase=6, n_sample=5, seed=0,
                  probe_cache=None, eval_frac=0.1):
    """Download, probe (or load cache), balance, split train/eval.
    Returns (train_instances, eval_instances, ood_pool, report)."""
    from build_dataset import build_drm_items
    from load_external import (load_triviaqa_facts, load_popqa_facts,
                               load_conflictqa, load_conflictqa_synthetic,
                               load_nq_ood)
    from probe_knowledge import probe_fact, apply_stability_label

    # 1. DRM controls (no probing)
    drm = build_drm_items(seed=seed)

    # 2. oversampled facts + conflict (with synthetic fallback for ConflictQA)
    #    TriviaQA is closed-book QA by design: probing splits it into PARAMETRIC
    #    (facts the model knows) and CONFABULATED (facts it doesn't). PopQA adds
    #    long-tail entities -> more CONFABULATED. (SQuAD was dropped: it's
    #    reading-comprehension, so its questions/answers assume a passage and
    #    don't work closed-book.)
    facts = load_triviaqa_facts(n_trivia, seed=seed) + \
            load_popqa_facts(n_popqa, seed=seed + 1)
    try:
        conflict = load_conflictqa(n_conflict, seed=seed + 2)
        if len(conflict) < max(50, n_conflict // 10):
            raise RuntimeError("ConflictQA returned too few items")
    except Exception as e:
        print(f"[assemble] ConflictQA unavailable ({e}); using synthetic conflicts")
        conflict = load_conflictqa_synthetic(n_conflict, seed=seed + 2)
    ood = load_nq_ood(2000, seed=seed + 3)   # held out, unprobed

    # 3. probe each fact on THIS model (model-relative labels)
    #    CACHE: if a probed file exists for this model, load it and skip probing.
    #    This saves hours on re-runs and makes results reproducible (same labels).
    import os
    from schema import save_dataset as _save, load_dataset as _load
    use_cache = False
    if probe_cache and os.path.exists(probe_cache):
        cached = _load(probe_cache)
        # guard: a cache much smaller than the requested pool is stale/partial
        # (e.g. left over from a small --n_trivia test run). Ignore and re-probe.
        if len(cached) >= 0.5 * len(facts):
            print(f"[assemble] loading cached probed facts <- {probe_cache} "
                  f"({len(cached)} items, skipping probing)")
            facts = cached
            use_cache = True
        else:
            print(f"[assemble] IGNORING cache {probe_cache}: only {len(cached)} "
                  f"items vs requested {len(facts)} (stale/partial). Re-probing.")
    if use_cache:
        pass
    else:
        print(f"[assemble] probing {len(facts)} fact candidates "
              f"({n_paraphrase} paraphrases x {n_sample} samples each)...")
        try:
            from tqdm import tqdm
            iterator = tqdm(facts, desc="probing", unit="fact", dynamic_ncols=True)
            use_tqdm = True
        except ImportError:
            iterator = facts
            use_tqdm = False
        counts = {"parametric": 0, "confabulated": 0, "gray": 0}
        for k, inst in enumerate(iterator):
            gold = inst.answer_aliases if inst.answer_aliases else inst.target
            rec = probe_fact(model, tokenizer, inst.question, gold,
                             n_paraphrase=n_paraphrase, n_sample=n_sample, device=device)
            apply_stability_label(inst, rec)
            cat = rec["category"]
            if cat == "prompt_agnostic":
                counts["parametric"] += 1
            elif cat == "unanswerable":
                counts["confabulated"] += 1
            else:
                counts["gray"] += 1
            if use_tqdm:
                iterator.set_postfix(param=counts["parametric"],
                                     confab=counts["confabulated"], gray=counts["gray"])
            elif (k + 1) % 100 == 0:
                print(f"[assemble]   probed {k+1}/{len(facts)}  "
                  f"param={counts['parametric']} confab={counts['confabulated']} "
                  f"gray={counts['gray']}")
        # save probed facts so future runs skip probing (reproducible labels)
        if probe_cache:
            os.makedirs(os.path.dirname(probe_cache), exist_ok=True)
            _save(facts, probe_cache)
            print(f"[assemble] cached probed facts -> {probe_cache}")

    # 4. balance to targets
    final, report = assemble_from_labeled(
        drm, facts, conflict,
        target_parametric=target_parametric, target_confab=target_confab,
        target_conflict=target_conflict, seed=seed)

    # 5. split into train / eval (held-out). Split DRM controls and non-DRM
    #    items SEPARATELY so the eval set always keeps enough DRM critical-lure
    #    probes (the confabulation ground-truth) rather than a random handful.
    import random as _random
    rng = _random.Random(seed)

    drm_items = [x for x in final if x.item_type == "drm"]
    other = [x for x in final if x.item_type != "drm"]
    rng.shuffle(drm_items)
    rng.shuffle(other)

    def _split(xs, frac):
        n = max(1, int(len(xs) * frac))
        return xs[n:], xs[:n]      # train, eval

    # DRM is limited (36 lists); use a higher eval fraction so the eval set has
    # enough critical-lure probes to estimate a confabulation rate.
    drm_tr, drm_ev = _split(drm_items, max(eval_frac, 0.3))
    oth_tr, oth_ev = _split(other, eval_frac)
    train_set = drm_tr + oth_tr
    eval_set = drm_ev + oth_ev
    rng.shuffle(train_set)
    rng.shuffle(eval_set)

    print(f"[assemble] report: {report}")
    print(f"[assemble] split: {len(train_set)} train / {len(eval_set)} eval "
          f"(eval_frac={eval_frac}); eval DRM={len(drm_ev)} non-DRM={len(oth_ev)}")
    return train_set, eval_set, ood, report


if __name__ == "__main__":
    # unit test of the balancing logic with synthetic labeled items
    def mk(src, i, conflict=0, resolved=True):
        s = src if resolved else "pending"
        lab = resolved
        return Instance(id=f"{src}_{i}", item_type="fact",
                        probe_kind="parametric_fact", context="", question="q",
                        target="a",
                        claims=[Claim(text="a", source=s, conflict=conflict,
                                      labeled=lab or (src != "pending"))])
    drm = [mk("verbatim", i) for i in range(50)]
    facts = ([mk("parametric", i) for i in range(4000)] +
             [mk("confabulated", i) for i in range(1200)] +
             [mk("pending", i, resolved=False) for i in range(500)])
    conflicts = [Instance(id=f"c_{i}", item_type="conflict", probe_kind="conflict",
                          context="ctx", question="q", target="a",
                          claims=[Claim(text="a", source="verbatim", conflict=1)])
                 for i in range(1500)]
    final, rep = assemble_from_labeled(drm, facts, conflicts)
    print("balancing report:", rep)
    print("final size:", len(final))
    from collections import Counter
    c = Counter(x.claims[0].source for x in final if x.claims[0].is_resolved())
    print("final source distribution:", dict(c))
