"""
pipeline.py
===========
End-to-end driver tying the modules together.

Stages:
  build     -> construct dataset, save JSONL
  measure   -> extract per-claim signals (real model OR mock), save features
  evaluate  -> compute the four metric groups from prediction records, save
  (train/probe are real-model stages; stubs documented here, run on the server)

Run `python pipeline.py --mock all` to exercise the full data/measure/analyze
path WITHOUT torch or a GPU, using synthetic signals + synthetic predictions.
This verifies the plumbing before you move to the server.
"""

from __future__ import annotations
import argparse
import json
import os
import numpy as np

from schema import load_dataset, save_dataset
from build_dataset import build_all
from signals import token_signals, attention_pool
from model import class_weights, total_loss_np, CLASS_TO_IDX
from analyze import evaluate_all, save_results, save_predictions, load_predictions
from checkpoint import save_config


# ---------------------------------------------------------------------------
# Synthetic signal generator: produces attention consistent with a claim's TRUE
# source, so we can validate that the measurement+analysis path recovers it.
# ---------------------------------------------------------------------------

def synth_attention(true_source: str, n_heads=8, n_q=10, n_k=10, ctx=4, rng=None):
    rng = rng or np.random.default_rng(0)
    attn = np.full((n_heads, n_q, n_k), 0.02)
    q = n_q - 1  # the claim token = last position
    if true_source == "verbatim":
        attn[:, q, rng.integers(0, ctx)] = 0.9         # focus on context
    elif true_source == "parametric":
        attn[:, q, rng.integers(ctx, n_k)] = 0.9        # focus, non-context
    else:  # confabulated: diffuse
        attn[:, q, ctx:] = 1.0 / n_k
        attn[:, q] += rng.uniform(0, 0.05, size=n_k)
    attn = attn / attn.sum(axis=2, keepdims=True)
    context_idx = np.arange(ctx)
    valid = np.arange(1, n_q + 1)
    return attn, context_idx, valid, q


def mock_measure(instances, out_path, seed=0):
    """Produce per-claim pooled signals u_j from synthetic attention."""
    rng = np.random.default_rng(seed)
    feats = []
    for inst in instances:
        for c in inst.claims:
            # include resolved claims (for attribution) AND unlabeled negative
            # controls (unrelated probes) which still need behavioral measurement
            is_neg_control = (inst.probe_kind == "unrelated")
            if not (c.is_resolved() or is_neg_control):
                continue
            # negative controls have no true source; synthesize as 'confabulated'
            # attention pattern only to drive the behavioral 'answered_yes' field
            src_for_attn = c.source if c.is_resolved() else "parametric"
            attn, cidx, valid, q = synth_attention(src_for_attn, rng=rng)
            z = token_signals(attn, cidx, valid_counts=valid)
            u = attention_pool(z, np.array([q]))
            feats.append({"instance_id": inst.id, "probe_kind": inst.probe_kind,
                          "item_type": inst.item_type,
                          "true_source": c.source if c.is_resolved() else None,
                          "conflict": c.conflict, "list_type": inst.list_type,
                          "u": u.tolist()})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for r in feats:
            f.write(json.dumps(r) + "\n")
    return feats


def mock_predictions(feats, seed=0):
    """Simple nearest-prototype 'classifier' on u to emulate a trained head,
    plus derived probe-specific fields for the DRM/conflict metrics."""
    rng = np.random.default_rng(seed)
    # prototypes in [context_mass, entropy, consensus] space
    proto = {"verbatim": np.array([0.9, 0.3, 1.0]),
             "parametric": np.array([0.1, 0.3, 1.0]),
             "confabulated": np.array([0.1, 0.9, 1.0])}
    recs = []
    for r in feats:
        u = np.array(r["u"])
        pred = min(proto, key=lambda c: np.linalg.norm(u - proto[c]))
        rec = {k: r[k] for k in ("instance_id", "probe_kind", "item_type",
                                 "true_source", "list_type")}
        rec["pred_source"] = pred
        # derived behavioral fields
        if r["probe_kind"] == "critical_lure":
            rec["answered_yes"] = (pred == "verbatim")  # said lure was present
        elif r["probe_kind"] == "studied":
            rec["answered_yes"] = (pred == "verbatim")
        elif r["probe_kind"] == "unrelated":
            rec["answered_yes"] = bool(rng.random() < 0.05)
        if r["probe_kind"] == "conflict":
            rec["followed_context"] = (pred == "verbatim")
        elif r["probe_kind"] == "non_conflict_control":
            rec["followed_context"] = (pred == "verbatim")
        rec["task_correct"] = bool(rng.random() < 0.85)
        recs.append(rec)
    return recs


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def stage_build(args):
    items = build_all(seed=args.seed)
    save_dataset(items, args.dataset)
    print(f"[build] {len(items)} instances -> {args.dataset}")


def stage_measure_mock(args):
    items = load_dataset(args.dataset)
    feats = mock_measure(items, args.features, seed=args.seed)
    print(f"[measure:mock] {len(feats)} claim features -> {args.features}")


def stage_evaluate_mock(args):
    # load features, fake predictions, evaluate, save
    feats = [json.loads(l) for l in open(args.features) if l.strip()]
    recs = mock_predictions(feats, seed=args.seed)
    save_predictions(recs, args.predictions)
    results = evaluate_all(recs)

    # also record class weights derived from the data (feeds real training)
    from collections import Counter
    counts = Counter(r["true_source"] for r in feats if r["true_source"])
    beta = class_weights(dict(counts))
    results["class_weights"] = dict(zip(["verbatim", "parametric", "confabulated"],
                                        [round(b, 4) for b in beta]))
    save_results(results, args.results)
    print(f"[evaluate:mock] predictions -> {args.predictions}")
    print(f"[evaluate:mock] results -> {args.results}")
    print(json.dumps(results, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["build", "measure", "evaluate", "all"])
    p.add_argument("--mock", action="store_true",
                   help="run measure/evaluate with synthetic signals (no torch)")
    p.add_argument("--dataset", default="data/dataset.jsonl")
    p.add_argument("--features", default="outputs/features.jsonl")
    p.add_argument("--predictions", default="outputs/predictions.jsonl")
    p.add_argument("--results", default="outputs/results.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.stage in ("build", "all"):
        stage_build(args)
    if args.stage in ("measure", "all"):
        if args.mock:
            stage_measure_mock(args)
        else:
            raise SystemExit("non-mock measure requires torch; run on server "
                             "(see signals.extract_attention_for_instance).")
    if args.stage in ("evaluate", "all"):
        if args.mock:
            stage_evaluate_mock(args)
        else:
            raise SystemExit("non-mock evaluate consumes real predictions.")


if __name__ == "__main__":
    main()
