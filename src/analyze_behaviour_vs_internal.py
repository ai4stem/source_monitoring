"""
analyze_behaviour_vs_internal.py  —  behavioural versus internal detection, like for like
=====================================================================================
All signals are scored on the SAME facts and the SAME label (correct vs incorrect), on a
class-balanced sample of each model's 6,000 facts:

  behavioural   self-consistency = LOPO support; entropy over answer clusters; each with
                answers compared by exact string and by meaning (anchor clustering);
                and the two meaning-level signals combined (logistic, 5-fold CV)
  internal      linear probe on the hidden state, 5-fold CV, best of four depths
  combined      logistic on [internal out-of-fold logit, support, entropy], 5-fold CV;
                the gain over behaviour alone is what the hidden state adds
  calibration   Brier score and ECE (10 bins) of the internal probe's out-of-fold
                probabilities, raw and after cross-fitted Platt scaling

Same probe settings as analyze_2x2*.py. CPU only.
OUTPUT  outputs/behaviour_vs_internal.json
Usage:  python analyze_behaviour_vs_internal.py --model all --jobs 13
"""
from __future__ import annotations
import argparse, glob, json, os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
from analyze_2x2_matched import _auroc
from analyze_2x2 import _balance


def cv_oof(X, y, k=5, seed=0, scale=True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(seed); idx = rng.permutation(len(X))
    folds = np.array_split(idx, k); oof = np.zeros(len(X))
    for i in range(k):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return oof


def ece(p, y, bins=10):
    conf = np.maximum(p, 1 - p); pred = (p >= 0.5).astype(int); acc = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def run_model(key):
    z = np.load(f"outputs/hidden2x2_{key}.npz"); s = np.load(f"outputs/support_semantic_{key}.npz")
    fracs = [float(f) for f in z["fracs"]]
    correct = z["correct"].astype(int)
    sel = _balance(np.where(correct == 0)[0], np.where(correct == 1)[0], seed=0)
    y = correct[sel]
    A = lambda v: round(float(_auroc(list(v), list(y))), 4)
    sup_m = s["support_anchor"].astype(float)[sel]; ent_m = s["entropy_anchor"].astype(float)[sel]
    sup_s = s["support_string"].astype(float)[sel]; ent_s = z["entropy"].astype(float)[sel]
    res = {"model": key, "n_per_class": int(len(sel) // 2),
           "sc_string": A(sup_s), "se_string": A(-ent_s), "sc_meaning": A(sup_m), "se_meaning": A(-ent_m)}
    B = np.column_stack([sup_m, ent_m])
    res["behaviour_combined"] = A(cv_oof(B, y))
    best, best_oof, best_f = -1, None, None
    for f in fracs:
        X = z[f"X_{f}"].astype(np.float32)[sel]
        fin = np.isfinite(X).all(axis=1)
        oof = np.full(len(sel), 0.5); oof[fin] = cv_oof(X[fin], y[fin])
        a = _auroc(list(oof[fin]), list(y[fin]))
        if a > best:
            best, best_oof, best_f = a, oof, f
    res["internal"] = round(float(best), 4); res["internal_depth"] = best_f
    p = np.clip(best_oof, 1e-6, 1 - 1e-6); logit = np.log(p / (1 - p))
    res["combined"] = A(cv_oof(np.column_stack([logit, sup_m, ent_m]), y))
    res["gain_over_behaviour"] = round(res["combined"] - res["behaviour_combined"], 4)
    res["brier_raw"] = round(float(np.mean((p - y) ** 2)), 4); res["ece_raw"] = round(ece(p, y), 4)
    pc = cv_oof(logit.reshape(-1, 1), y, seed=1)
    res["brier_platt"] = round(float(np.mean((pc - y) ** 2)), 4); res["ece_platt"] = round(ece(pc, y), 4)
    print(f"[done] {key}: SC {res['sc_meaning']} | internal {res['internal']} | combined {res['combined']} "
          f"(behaviour {res['behaviour_combined']})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all"); ap.add_argument("--jobs", type=int, default=13)
    ap.add_argument("--out", default="outputs/behaviour_vs_internal.json")
    args = ap.parse_args()
    keys = sorted(os.path.basename(p)[10:-4] for p in glob.glob("outputs/hidden2x2_*.npz")) \
        if args.model == "all" else args.model.split(",")
    from multiprocessing import Pool
    with Pool(min(args.jobs, len(keys))) as pool:
        rows = pool.map(run_model, keys)
    json.dump(rows, open(args.out, "w"), indent=2)
    import statistics as st
    for k in ["sc_string", "se_string", "sc_meaning", "se_meaning", "behaviour_combined", "internal", "combined",
              "gain_over_behaviour", "brier_raw", "ece_raw", "brier_platt", "ece_platt"]:
        v = [r[k] for r in rows]
        print(f"{k:20s} mean {st.mean(v):.3f}  range {min(v):.3f}-{max(v):.3f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
