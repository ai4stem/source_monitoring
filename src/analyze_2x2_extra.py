"""
analyze_2x2_extra.py  —  three additional controls for the primary matched analysis
=====================================================================================
All on the primary support definition (anchor) and the same probe as analyze_2x2_primary.

  A  PopQA question properties.  PopQA records, for every question, the relation it asks
     about (16 templates), and the popularity (monthly Wikipedia page views) of the
     subject and of the answer entity. Within PopQA, correct/incorrect pairs are formed
       support            caliper 0.05 on support                    (as in the primary)
       +relation          same relation template
       +subject pop.      and |log10 s_pop difference| <= 0.25
       +answer pop.       and |log10 o_pop difference| <= 0.25
     and the hidden-state correctness AUROC is reported for each, with the AUROC of each
     scalar alone on the pairs. "strict" joins the TriviaQA pairs of the primary analysis
     with the +relation+subject-popularity PopQA pairs.
  B  Non-linear probe (one hidden layer, 256 units) on the primary matched pairs, at the
     best linear depth, same folds.
  C  Permuted labels: the full procedure (best of four depths, 5-fold CV) with the
     correct/incorrect labels of the primary matched pairs shuffled, 5 permutations.

CPU. OUTPUT outputs/analysis_2x2_extra.json
"""
from __future__ import annotations
import glob, json, os
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import numpy as np
from analyze_2x2_matched import caliper_match, _auroc, _cv_auroc
from analyze_2x2_robustness import _fit_predict, _folds

POP_CAL = 0.25


def fact_meta(key, n, pop):
    ids, qs = [], []
    for line in open(f"outputs/probed_{key}.jsonl"):
        x = json.loads(line)
        if x.get("item_type") != "fact" or not (x.get("stability") or {}).get("by_paraphrase"):
            continue
        ids.append(x["id"]); qs.append(x.get("question") or "")
    ids, qs = ids[:n], qs[:n]
    ds = np.array([0 if i.startswith("trivia") else 1 for i in ids])
    rel = np.full(n, -1); sp = np.full(n, np.nan); op = np.full(n, np.nan)
    for j, (i, q) in enumerate(zip(ids, qs)):
        if ds[j] == 1:
            r = pop[int(i.split("_")[1])]
            assert r["question"].strip() == q.strip(), (i, q, r["question"])
            rel[j] = r["prop_id"]; sp[j] = np.log10(max(r["s_pop"], 1)); op[j] = np.log10(max(r["o_pop"], 1))
    return ds, rel, sp, op


def multi_match(ci, ii, support, strata, extra, caliper, seed=0):
    """Greedy 1:1 matching of incorrect to correct facts: same stratum, support within
    caliper, every extra covariate within POP_CAL. Nearest by scaled distance."""
    rng = np.random.RandomState(seed); mp, mn = [], []
    for st in np.unique(strata[np.concatenate([ci, ii])]):
        c = ci[strata[ci] == st]; i_ = ii[strata[ii] == st]
        if not len(c) or not len(i_):
            continue
        free = np.ones(len(c), bool)
        for j in rng.permutation(len(i_)):
            a = i_[j]
            dist = np.abs(support[c] - support[a]) / caliper
            ok = free & (dist <= 1 + 1e-9)
            for e in extra:
                de = np.abs(e[c] - e[a]) / POP_CAL
                ok &= de <= 1 + 1e-9; dist = dist + de
            if ok.any():
                k = np.where(ok)[0][np.argmin(dist[ok])]
                free[k] = False; mp.append(c[k]); mn.append(a)
    return np.array(mp, int), np.array(mn, int)


def run_model(args):
    key, pop = args
    z = np.load(f"outputs/hidden2x2_{key}.npz"); s = np.load(f"outputs/support_semantic_{key}.npz")
    fracs = [float(f) for f in z["fracs"]]
    X = {f: z[f"X_{f}"].astype(np.float32) for f in fracs}
    correct = z["correct"].astype(int); support = s["support_anchor"].astype(float); n = len(correct)
    ds, rel, sp, op = fact_meta(key, n, pop)
    res = {"model": key}

    def score(mp, mn, scalars=()):
        out = {"n_pairs": int(len(mp)), "truth_AUROC": None}
        if len(mp) < 40:
            return out
        gi = np.concatenate([mp, mn]); y = np.array([1] * len(mp) + [0] * len(mn))
        per = {f: _cv_auroc(X[f][gi], y, seed=0) for f in fracs}
        out["truth_AUROC"] = round(float(max(per.values())), 4)
        out["support_only"] = round(float(_auroc(list(support[gi]), list(y))), 4)
        for name, v in scalars:
            out[name + "_only"] = round(float(_auroc(list(v[gi]), list(y))), 4)
        return out

    # ---- A. PopQA
    pc = np.where((correct == 1) & (ds == 1))[0]; pi = np.where((correct == 0) & (ds == 1))[0]
    one = np.zeros(n, int); sc = (("subject_pop", sp), ("answer_pop", op))
    A = {"n_correct": int(len(pc)), "n_incorrect": int(len(pi)),
         "pop_only_all": {"subject": round(float(_auroc(list(sp[ds == 1]), list(correct[ds == 1]))), 4),
                          "answer": round(float(_auroc(list(op[ds == 1]), list(correct[ds == 1]))), 4)}}
    # how well does the relation alone predict correctness on support-matched PopQA pairs?
    m0 = multi_match(pc, pi, support, one, [], 0.05)
    gi = np.concatenate(m0); y = np.array([1] * len(m0[0]) + [0] * len(m0[1]))
    rate = {r: correct[(ds == 1) & (rel == r)].mean() for r in np.unique(rel[ds == 1])}
    A["support"] = score(*m0, sc)
    A["support"]["relation_only"] = round(float(_auroc([rate[r] for r in rel[gi]], list(y))), 4)
    m1 = multi_match(pc, pi, support, rel, [], 0.05); A["support_relation"] = score(*m1, sc)
    m2 = multi_match(pc, pi, support, rel, [sp], 0.05); A["support_relation_spop"] = score(*m2, sc)
    m3 = multi_match(pc, pi, support, rel, [sp, op], 0.05); A["support_relation_spop_opop"] = score(*m3, sc)
    res["popqa"] = A
    # ---- primary matched pairs (within dataset, support caliper), as in analyze_2x2_primary
    prim = []
    for v in (0, 1):
        ci = np.where((correct == 1) & (ds == v))[0]; ii = np.where((correct == 0) & (ds == v))[0]
        prim.append(caliper_match(support[ci], ci, support[ii], ii, 0.05, seed=0))
    tp, tn = prim[0]
    res["strict"] = score(np.concatenate([tp, m2[0]]), np.concatenate([tn, m2[1]]))
    mp = np.concatenate([prim[0][0], prim[1][0]]); mn = np.concatenate([prim[0][1], prim[1][1]])
    gi = np.concatenate([mp, mn]); y = np.array([1] * len(mp) + [0] * len(mn))
    per = {f: _cv_auroc(X[f][gi], y, seed=0) for f in fracs}; bf = max(per, key=per.get)
    # ---- B. MLP on the primary pairs
    folds = _folds(len(gi), 5, 0); oof = np.zeros(len(gi))
    for i in range(5):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(5) if j != i])
        oof[te] = _fit_predict(X[bf][gi][tr], y[tr], X[bf][gi][te], kind="mlp")
    res["primary_linear"] = round(float(per[bf]), 4)
    res["primary_mlp"] = round(float(_auroc(list(oof), list(y))), 4)
    # ---- C. permuted labels
    perm = []
    for r in range(5):
        yp = np.random.RandomState(100 + r).permutation(y)
        perm.append(max(_cv_auroc(X[f][gi], yp, seed=0) for f in fracs))
    res["permuted"] = {"values": [round(float(v), 4) for v in perm], "mean": round(float(np.mean(perm)), 4)}
    print(f"[done] {key}: popqa support {A['support']['truth_AUROC']} (rel-only {A['support']['relation_only']}) "
          f"+rel {A['support_relation']['truth_AUROC']} (n={A['support_relation']['n_pairs']}) "
          f"+spop {A['support_relation_spop']['truth_AUROC']} (n={A['support_relation_spop']['n_pairs']}) "
          f"+opop {A['support_relation_spop_opop']['truth_AUROC']} (n={A['support_relation_spop_opop']['n_pairs']}) | "
          f"strict {res['strict']['truth_AUROC']} | lin {res['primary_linear']} mlp {res['primary_mlp']} | perm {res['permuted']['mean']}", flush=True)
    return res


if __name__ == "__main__":
    from datasets import load_dataset
    from multiprocessing import Pool
    d = load_dataset("akariasai/PopQA", split="test")
    pop = [{"question": q, "prop_id": p, "s_pop": a, "o_pop": b}
           for q, p, a, b in zip(d["question"], d["prop_id"], d["s_pop"], d["o_pop"])]
    keys = sorted(os.path.basename(p)[10:-4] for p in glob.glob("outputs/hidden2x2_*.npz"))
    with Pool(13) as pool:
        rows = pool.map(run_model, [(k, pop) for k in keys])
    json.dump(rows, open("outputs/analysis_2x2_extra.json", "w"), indent=2)
    print("saved -> outputs/analysis_2x2_extra.json")
