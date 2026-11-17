"""analyze_behaviour_dataset.py — does the hidden state's increment over behaviour survive
when the detector is also told which dataset the question came from?
Same facts, labels, probe and CV as analyze_behaviour_vs_internal.py; adds the dataset
indicator (TriviaQA / PopQA) to the behavioural detector, and repeats the comparison
within TriviaQA only. CPU. OUTPUT outputs/behaviour_dataset.json"""
import glob, json, os
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
from analyze_2x2_matched import _auroc
from analyze_2x2 import _balance
from analyze_2x2_dataset import dataset_ids
from analyze_behaviour_vs_internal import cv_oof


def block(z, s, fracs, sel, y, ds=None):
    A = lambda v: round(float(_auroc(list(v), list(y))), 4)
    sup = s["support_anchor"].astype(float)[sel]; ent = s["entropy_anchor"].astype(float)[sel]
    best, best_oof = -1, None
    for f in fracs:
        X = z[f"X_{f}"].astype(np.float32)[sel]
        oof = cv_oof(X, y); a = _auroc(list(oof), list(y))
        if a > best:
            best, best_oof = a, oof
    p = np.clip(best_oof, 1e-6, 1 - 1e-6); logit = np.log(p / (1 - p))
    out = {"n_per_class": int(len(sel) // 2), "sc": A(sup), "internal": round(float(best), 4),
           "behaviour": A(cv_oof(np.column_stack([sup, ent]), y)),
           "behaviour_internal": A(cv_oof(np.column_stack([logit, sup, ent]), y))}
    out["gain"] = round(out["behaviour_internal"] - out["behaviour"], 4)
    if ds is not None:
        d = ds[sel].astype(float)
        out["dataset_only"] = A(-d)
        out["behaviour_dataset"] = A(cv_oof(np.column_stack([sup, ent, d]), y))
        out["behaviour_dataset_internal"] = A(cv_oof(np.column_stack([logit, sup, ent, d]), y))
        out["gain_given_dataset"] = round(out["behaviour_dataset_internal"] - out["behaviour_dataset"], 4)
    return out


def run_model(key):
    z = np.load(f"outputs/hidden2x2_{key}.npz"); s = np.load(f"outputs/support_semantic_{key}.npz")
    fracs = [float(f) for f in z["fracs"]]; correct = z["correct"].astype(int); ds = dataset_ids(key, len(correct))
    sel = _balance(np.where(correct == 0)[0], np.where(correct == 1)[0], seed=0)
    res = {"model": key, "all": block(z, s, fracs, sel, correct[sel], ds)}
    for name, v in (("trivia", 0), ("popqa", 1)):
        m = ds == v
        sel = _balance(np.where((correct == 0) & m)[0], np.where((correct == 1) & m)[0], seed=0)
        res[name] = block(z, s, fracs, sel, correct[sel])
    print(f"[done] {key}: gain {res['all']['gain']} | given dataset {res['all']['gain_given_dataset']} | "
          f"trivia {res['trivia']['gain']} | popqa {res['popqa']['gain']}", flush=True)
    return res


if __name__ == "__main__":
    from multiprocessing import Pool
    keys = sorted(os.path.basename(p)[10:-4] for p in glob.glob("outputs/hidden2x2_*.npz"))
    with Pool(13) as pool:
        rows = pool.map(run_model, keys)
    json.dump(rows, open("outputs/behaviour_dataset.json", "w"), indent=2)
    import statistics as st
    for blk, ks in (("all", ["sc", "internal", "dataset_only", "behaviour", "behaviour_internal", "gain", "behaviour_dataset",
                             "behaviour_dataset_internal", "gain_given_dataset"]),
                    ("trivia", ["sc", "internal", "behaviour", "behaviour_internal", "gain"]),
                    ("popqa", ["sc", "internal", "behaviour", "behaviour_internal", "gain"])):
        for k in ks:
            v = [r[blk][k] for r in rows]
            print(f"{blk:7s}{k:28s} mean {st.mean(v):.3f}  range {min(v):.3f} to {max(v):.3f}  n>0: {sum(x > 0 for x in v)}")
