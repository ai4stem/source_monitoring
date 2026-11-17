"""
test_core.py
============
Torch-free unit tests for the core logic. Run: python test_core.py
Validates the maths and plumbing that must be correct before moving to the GPU.
"""
import numpy as np
from schema import Instance, Claim, save_dataset, load_dataset, dataset_summary
from signals import (context_mass, normalized_entropy, cross_head_consensus,
                     token_signals, attention_pool)
from model import class_weights, source_loss_np, conflict_loss_np, CLASS_TO_IDX
from analyze import attribution_metrics, drm_confabulation_rate, conflict_metrics
from probe_knowledge import answer_entropy, classify_stability, stability_record


def test_schema_roundtrip(tmp="/tmp/_sm_test.jsonl"):
    insts = [Instance(id="x", item_type="drm", probe_kind="critical_lure",
                      context="list: a, b, c.", question="was 'd' present?",
                      target="no",
                      claims=[Claim(text="'d' present", source="confabulated")])]
    save_dataset(insts, tmp)
    back = load_dataset(tmp)
    assert back[0].id == "x" and back[0].claims[0].source == "confabulated"
    print("ok  schema roundtrip")


def test_signals_separation():
    n_heads, n_q, n_k, ctx = 8, 6, 6, 3
    attn = np.full((n_heads, n_q, n_k), 0.02)
    attn[:, 3, 1] = 0.9          # verbatim: context focus
    attn[:, 4, 4] = 0.9          # parametric: non-context focus
    attn[:, 5, ctx:] = 1.0/n_k   # confabulated: diffuse
    attn = attn/attn.sum(2, keepdims=True)
    z = token_signals(attn, np.arange(ctx), valid_counts=np.arange(1, n_q+1))
    # verbatim has highest context mass
    assert z[3, 0] > z[4, 0] and z[3, 0] > z[5, 0]
    # confabulated has highest entropy
    assert z[5, 1] > z[3, 1] and z[5, 1] > z[4, 1]
    print("ok  signal separation (m gates verbatim, H gates confab)")


def test_class_weights():
    beta = class_weights({"verbatim": 100, "parametric": 200, "confabulated": 20})
    assert beta[2] > beta[0] > beta[1]   # rare class upweighted
    assert abs(beta.sum() - 3.0) < 1e-6
    print("ok  class weights (effective number)")


def test_losses():
    logits = np.array([[3., 0., 0.], [0., 3., 0.], [2., 0., .5]])
    labels = np.array([0, 1, 2])
    beta = np.ones(3)
    l = source_loss_np(logits, labels, beta)
    assert l > 0
    # conflict claim leaning verbatim -> low; leaning parametric -> high
    lo = conflict_loss_np(np.array([[3., 0., 0.]]), np.array([True]))
    hi = conflict_loss_np(np.array([[0., 3., 0.]]), np.array([True]))
    assert hi > lo
    print("ok  losses (source CE, conflict pushes verbatim)")


def test_metrics():
    recs = [
        {"probe_kind": "critical_lure", "true_source": "confabulated",
         "pred_source": "confabulated", "answered_yes": True, "list_type": "high_BAS"},
        {"probe_kind": "studied", "true_source": "verbatim",
         "pred_source": "verbatim", "answered_yes": True},
        {"probe_kind": "unrelated", "answered_yes": False},
        {"probe_kind": "conflict", "true_source": "verbatim",
         "pred_source": "verbatim", "followed_context": True},
        {"probe_kind": "non_conflict_control", "followed_context": False},
    ]
    a = attribution_metrics(recs)
    assert a["macro_f1"] == 1.0
    d = drm_confabulation_rate(recs)
    assert d["critical_lure_false_yes_rate"] == 1.0
    assert d["unrelated_false_yes_rate"] == 0.0
    assert d["drm_effect_present"] is True
    c = conflict_metrics(recs)
    assert c["verbatim_override_rate"] == 1.0 and c["over_deference_rate"] == 0.0
    print("ok  metrics (attribution / DRM / conflict)")


def test_stability():
    # stable correct answers -> parametric
    r = stability_record(["Paris"]*8, [True]*8)
    assert r["category"] == "prompt_agnostic"
    # scattered wrong -> unanswerable (confabulation candidate)
    r2 = stability_record(["1783", "1791", "1802", "1776"], [False]*4)
    assert r2["category"] == "unanswerable"
    # middling correctness (2/5 = 0.4) -> gray zone
    r3 = stability_record(["a", "b", "a", "c", "d"], [True, False, True, False, False])
    assert r3["category"] == "prompt_sensitive", r3
    print("ok  stability classification (3-way knowledge boundary)")


if __name__ == "__main__":
    test_schema_roundtrip()
    test_signals_separation()
    test_class_weights()
    test_losses()
    test_metrics()
    test_stability()
    print("\nAll core tests passed.")
