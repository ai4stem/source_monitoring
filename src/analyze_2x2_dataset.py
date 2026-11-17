"""
analyze_2x2_dataset.py  —  is the matched correctness signal a dataset / question-type signal?
=====================================================================================
The 6,000 facts mix TriviaQA (4,000) and PopQA (2,000). The two differ in difficulty and
in question form, and a hidden state can certainly tell them apart. Matching on support
does not control for this. Here we

  * report accuracy by dataset and how well the dataset indicator alone predicts
    correctness on the support-matched pairs (should be 0.5 if there is no leak);
  * re-run the matched analysis with pairs formed WITHIN dataset (support caliper 0.05
    and same dataset), pooled, and separately for each dataset.

Same hidden states, probe and matching code as the other scripts. Anchor support. CPU only.
OUTPUT  outputs/analysis_2x2_dataset.json
"""
from __future__ import annotations
import argparse, glob, json, os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
from analyze_2x2_matched import caliper_match, hidden_auroc_over_depth, _auroc


def dataset_ids(key, n):
    out = []
    for line in open(f"outputs/probed_{key}.jsonl"):
        x = json.loads(line)
        if x.get("item_type") != "fact" or not (x.get("stability") or {}).get("by_paraphrase"):
            continue
        out.append(0 if x["id"].startswith("trivia") else 1)
    return np.array(out[:n])


def run_model(key, caliper=0.05):
    z = np.load(f"outputs/hidden2x2_{key}.npz"); s = np.load(f"outputs/support_semantic_{key}.npz")
    fracs = [float(f) for f in z["fracs"]]
    d = {f"X_{f}": z[f"X_{f}"] for f in fracs}
    correct = z["correct"].astype(int); support = s["support_anchor"].astype(float)
    ds = dataset_ids(key, len(correct)); assert len(ds) == len(correct)
    res = {"model": key, "accuracy_trivia": round(float(correct[ds == 0].mean()), 4),
           "accuracy_popqa": round(float(correct[ds == 1].mean()), 4)}

    def match(mask):
        ci = np.where((correct == 1) & mask)[0]; ii = np.where((correct == 0) & mask)[0]
        return caliper_match(support[ci], ci, support[ii], ii, caliper)

    mp, mn = match(np.ones(len(correct), bool))
    gi = np.concatenate([mp, mn]); y = np.array([1] * len(mp) + [0] * len(mn))
    res["pooled_matching"] = {"n_pairs": int(len(mp)),
                              "dataset_only_AUROC": round(float(_auroc(list(-ds[gi].astype(float)), list(y))), 4),
                              "share_popqa_correct": round(float(ds[mp].mean()), 4),
                              "share_popqa_incorrect": round(float(ds[mn].mean()), 4)}
    parts = {}
    for name, v in (("trivia", 0), ("popqa", 1)):
        p, n_ = match(ds == v); parts[name] = (p, n_)
        g = np.concatenate([p, n_]); yy = np.array([1] * len(p) + [0] * len(n_))
        res[f"within_{name}"] = {"n_pairs": int(len(p)),
                                 "truth_AUROC": hidden_auroc_over_depth(d, fracs, g, yy) if len(p) >= 60 else None}
    p = np.concatenate([parts["trivia"][0], parts["popqa"][0]]); n_ = np.concatenate([parts["trivia"][1], parts["popqa"][1]])
    g = np.concatenate([p, n_]); yy = np.array([1] * len(p) + [0] * len(n_))
    res["within_dataset_pooled"] = {"n_pairs": int(len(p)), "truth_AUROC": hidden_auroc_over_depth(d, fracs, g, yy),
                                    "dataset_only_AUROC": round(float(_auroc(list(-ds[g].astype(float)), list(yy))), 4),
                                    "support_only_AUROC": round(float(_auroc(list(support[g]), list(yy))), 4)}
    print(f"[done] {key}: acc T/P {res['accuracy_trivia']}/{res['accuracy_popqa']} | dataset-only on matched "
          f"{res['pooled_matching']['dataset_only_AUROC']} | within-dataset pooled {res['within_dataset_pooled']['truth_AUROC']} "
          f"| trivia {res['within_trivia']['truth_AUROC']} popqa {res['within_popqa']['truth_AUROC']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=13)
    ap.add_argument("--out", default="outputs/analysis_2x2_dataset.json")
    args = ap.parse_args()
    keys = sorted(os.path.basename(p)[10:-4] for p in glob.glob("outputs/hidden2x2_*.npz"))
    from multiprocessing import Pool
    with Pool(min(args.jobs, len(keys))) as pool:
        rows = pool.map(run_model, keys)
    json.dump(rows, open(args.out, "w"), indent=2)
    m = lambda f: float(np.mean([f(r) for r in rows if f(r) is not None]))
    print(f"\nmean: acc trivia {m(lambda r: r['accuracy_trivia']):.3f} popqa {m(lambda r: r['accuracy_popqa']):.3f} | "
          f"dataset-only on matched {m(lambda r: r['pooled_matching']['dataset_only_AUROC']):.3f} | "
          f"within-dataset pooled {m(lambda r: r['within_dataset_pooled']['truth_AUROC']):.3f} | "
          f"trivia {m(lambda r: r['within_trivia']['truth_AUROC']):.3f} | popqa {m(lambda r: r['within_popqa']['truth_AUROC']):.3f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
