"""summarize_nli.py — table of the primary analysis under every support definition."""
import json, statistics as st, numpy as np, sys
from analyze_2x2_matched import _auroc
rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "outputs/analysis_2x2_nli.json"))
V = ["string", "anchor", "gold", "nli_raw_strict", "nli_raw_loose", "nli_canon_strict", "nli_canon_loose"]
KEY = {"string": "support_string", "anchor": "support_anchor", "gold": "support_gold"}
def hm_lo(a, n):
    q1 = a / (2 - a); q2 = 2 * a * a / (1 + a)
    se = ((a * (1 - a) + (n - 1) * (q1 - a * a) + (n - 1) * (q2 - a * a)) / (n * n)) ** 0.5
    return a - 1.96 * se
print(f"{'variant':17s} {'sup-only':>8s} {'matched':>8s} {'range':>13s} {'CI>.5':>5s} {'pairs':>11s} {'trivia':>7s} {'rest':>6s} {'unan':>6s} {'n_un':>9s} {'una<rest':>8s}")
for v in V:
    so = []
    for r in rows:
        k = r["model"]; s = dict(np.load(f"outputs/support_semantic_{k}.npz")); s.update(np.load(f"outputs/support_nli_{k}.npz"))
        key = KEY.get(v) or "support_" + v[4:]
        so.append(_auroc(list(s[key].astype(float)), list(s["correct"].astype(int))))
    m = [r[v]["matched"]["truth_AUROC"] for r in rows]; n = [r[v]["matched"]["n_pairs"] for r in rows]
    tr = [r[v]["matched_trivia"]["truth_AUROC"] for r in rows]
    pr = [(r[v]["matched_rest"]["truth_AUROC"], r[v]["unanimous"]["truth_AUROC"]) for r in rows]
    pr2 = [p for p in pr if p[1] is not None]
    nu = [r[v]["unanimous"]["n_pairs"] for r in rows]
    sig = sum(hm_lo(a, k) > 0.5 for a, k in zip(m, n))
    print(f"{v:17s} {st.mean(so):8.3f} {st.mean(m):8.3f} {min(m):.3f}-{max(m):.3f} {sig:5d} {min(n):5d}-{max(n):5d} {st.mean(tr):7.3f} "
          f"{st.mean(p[0] for p in pr2) if pr2 else float('nan'):6.3f} {st.mean(p[1] for p in pr2) if pr2 else float('nan'):6.3f} "
          f"{min(nu):4d}-{max(nu):4d} {sum(p[1] < p[0] for p in pr2):3d}/{len(pr2)}")
