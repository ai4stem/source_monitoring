"""
probe_hidden.py  (SERVER-SIDE, fast — reads saved arrays)
=========================================================
On the hidden states saved by extract_hidden.py, run three analyses per model,
all on held-out folds, with NO generation:

  1. LINEAR probe  — logistic regression (our main-text result).
  2. MLP probe     — small nonlinear net; preempts "you only tried linear".
  3. CALIBRATION   — if the probe's confidence were used as a truthfulness
                     signal, how well is it calibrated? Report ECE and Brier
                     for the better probe, plus the base rate. This grounds the
                     metacognition/calibration discussion (ECE, Brier) without
                     over-claiming: it shows the ceiling a representation-based
                     confidence signal could reach.

For each model and the best depth fraction we report:
  lin_acc, mlp_acc  (balanced CV accuracy)
  lin_net, mlp_net  (acc - shuffled-label baseline)
  brier, ece        (cross-validated, from the better probe's held-out probs)

Usage:
    python probe_hidden.py --model all
    python probe_hidden.py --model qwen3-8b --verbose
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
from registry import get_model, MODELS


# ---------- metrics ----------
def _ece(probs, labels, n_bins=10):
    """Expected Calibration Error."""
    probs = np.asarray(probs); labels = np.asarray(labels)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (probs >= bins[i]) & (probs < bins[i + 1] if i < n_bins - 1
                                  else probs <= bins[i + 1])
        if m.sum() == 0:
            continue
        conf = probs[m].mean()
        acc = labels[m].mean()
        ece += (m.sum() / len(probs)) * abs(conf - acc)
    return float(ece)


def _brier(probs, labels):
    probs = np.asarray(probs); labels = np.asarray(labels)
    return float(np.mean((probs - labels) ** 2))


def _balance(X, y, seed=0):
    y = np.asarray(y)
    i0 = np.where(y == 0)[0]; i1 = np.where(y == 1)[0]
    m = min(len(i0), len(i1))
    if m < 10:
        return None, None, m
    rng = np.random.RandomState(seed)
    sel = np.concatenate([rng.choice(i0, m, False), rng.choice(i1, m, False)])
    return X[sel], y[sel], m


def _cv_probs(clf_factory, X, y, k=5, seed=0):
    """K-fold; return out-of-fold predicted prob for class 1 and accuracy."""
    from sklearn.preprocessing import StandardScaler
    n = len(X)
    rng = np.random.RandomState(seed)
    folds = np.array_split(rng.permutation(n), k)
    oof = np.zeros(n)
    for i in range(k):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        sc = StandardScaler().fit(X[tr])
        clf = clf_factory()
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    acc = float(((oof > 0.5).astype(int) == y).mean())
    return oof, acc


def _numpy_logreg_factory():
    """Fallback linear classifier with predict_proba, if sklearn missing."""
    class _LR:
        def fit(self, X, y):
            Xb = np.hstack([X, np.ones((len(X), 1))])
            yb = y * 2.0 - 1
            self.w = np.linalg.lstsq(Xb.T @ Xb + 1e-1 * np.eye(Xb.shape[1]),
                                     Xb.T @ yb, rcond=None)[0]
            return self
        def predict_proba(self, X):
            Xb = np.hstack([X, np.ones((len(X), 1))])
            s = 1 / (1 + np.exp(-(Xb @ self.w)))
            return np.stack([1 - s, s], axis=1)
    return _LR()


def probe_model(key, verbose=False):
    fn = f"outputs/hidden_{key}.npz"
    if not os.path.exists(fn):
        return None
    d = np.load(fn)
    y_all = d["y"].astype(int)
    fracs = list(d["fracs"])
    spec = get_model(key)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        have_sklearn = True
    except ImportError:
        have_sklearn = False

    best = {"net": -1, "frac": None}
    for f in fracs:
        X = d[f"X_{f}"].astype(np.float32)
        # some models (e.g. gemma) have large activations that overflow float16
        # to inf on save; drop non-finite rows so the scaler/probe don't crash.
        finite = np.isfinite(X).all(axis=1)
        if not finite.all():
            X = X[finite]
            y_f = y_all[finite]
        else:
            y_f = y_all
        Xb, yb, m = _balance(X, y_f)
        if Xb is None:
            continue
        # linear
        if have_sklearn:
            lin_fac = lambda: LogisticRegression(max_iter=1000, C=1.0)
        else:
            lin_fac = _numpy_logreg_factory
        _, lin_acc = _cv_probs(lin_fac, Xb, yb)
        # shuffled baseline
        rng = np.random.RandomState(1)
        _, lin_base = _cv_probs(lin_fac, Xb, rng.permutation(yb))
        lin_net = lin_acc - lin_base

        # MLP (only if sklearn present)
        if have_sklearn:
            mlp_fac = lambda: MLPClassifier(hidden_layer_sizes=(128,),
                                            max_iter=300, early_stopping=True,
                                            random_state=0)
            mlp_oof, mlp_acc = _cv_probs(mlp_fac, Xb, yb)
            _, mlp_base = _cv_probs(mlp_fac, Xb, rng.permutation(yb))
            mlp_net = mlp_acc - mlp_base
        else:
            mlp_oof, mlp_acc, mlp_net = None, None, None

        # pick better probe for calibration at this frac
        use_oof, use_acc = (mlp_oof, mlp_acc) if (mlp_acc is not None and
                                                  mlp_acc >= lin_acc) else (None, lin_acc)
        if use_oof is None:
            use_oof, _ = _cv_probs(lin_fac, Xb, yb)
        brier = _brier(use_oof, yb)
        ece = _ece(use_oof, yb)

        cur_net = max(lin_net, mlp_net if mlp_net is not None else -1)
        if verbose:
            mlp_a = "n/a" if mlp_acc is None else f"{round(mlp_acc, 3)}"
            mlp_n = "n/a" if mlp_net is None else f"{round(mlp_net, 3)}"
            print(f"  frac={f:.2f} lin_acc={lin_acc:.3f}(net {lin_net:+.3f}) "
                  f"mlp_acc={mlp_a}(net {mlp_n}) "
                  f"brier={brier:.3f} ece={ece:.3f}")
        if cur_net > best["net"]:
            best = {"net": cur_net, "frac": f, "lin_acc": lin_acc, "lin_net": lin_net,
                    "mlp_acc": mlp_acc, "mlp_net": mlp_net, "brier": brier,
                    "ece": ece, "n_per_class": m}
    if best["frac"] is None:
        return None
    best.update({"model": key, "size_b": spec.size_b})
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all")
    p.add_argument("--out", default="outputs/probe_hidden.json")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    keys = [m.key for m in MODELS if m.access == "weights"] \
        if args.model == "all" else [args.model]

    rows = []
    for k in keys:
        if not os.path.exists(f"outputs/hidden_{k}.npz"):
            print(f"[skip] {k}: no hidden_{k}.npz (run extract_hidden.py first)")
            continue
        print(f"[probe] {k}")
        r = probe_model(k, verbose=args.verbose)
        if r:
            rows.append(r)
            mlp_disp = "n/a" if r["mlp_net"] is None else f"{round(r['mlp_net'], 3)}"
            print(f"   best frac={r['frac']} | lin_net={r['lin_net']:+.3f} "
                  f"mlp_net={mlp_disp} | brier={r['brier']:.3f} ece={r['ece']:.3f}")

    rows.sort(key=lambda r: r["size_b"])
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)

    # summary: does MLP beat linear? (the robustness question)
    print(f"\n{'model':18s} {'size':>5s} {'lin_net':>8s} {'mlp_net':>8s} "
          f"{'Δ(mlp-lin)':>10s} {'brier':>7s} {'ece':>6s}")
    print("-" * 66)
    for r in rows:
        dl = (r["mlp_net"] - r["lin_net"]) if r["mlp_net"] is not None else float("nan")
        mlp_str = "n/a" if r["mlp_net"] is None else f"{r['mlp_net']:+.3f}"
        print(f"{r['model']:18s} {r['size_b']:>5.1f} {r['lin_net']:>+8.3f} "
              f"{mlp_str:>8} {dl:>+10.3f} {r['brier']:>7.3f} {r['ece']:>6.3f}")
    print("\nΔ(mlp-lin) near 0 => nonlinear probes do NOT rescue the signal "
          "(linear finding is robust).")
    print("brier/ece = calibration ceiling of a representation-based truth signal.")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
