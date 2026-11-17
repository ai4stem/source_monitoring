"""cluster_diagnostics.py — how often two answers to the same question share a cluster,
by correctness of the pair, for the rule-based clusterings (string / anchor / gold).
Comparable with the diagnostics stored by nli_cluster_support.py.
A good clustering puts two correct answers together (high) and a correct and an
incorrect answer apart (low). OUTPUT outputs/cluster_diagnostics.json"""
import glob, json, os
from multiprocessing import Pool
from recompute_support_semantic import _norm, cluster
from nli_cluster_support import load_facts, _diagnostics


def run(key):
    facts = load_facts(key)[:6000]
    out = {"model": key}
    labs = {"string": [], "anchor": [], "gold": []}
    for q, raws, flags in facts:
        n = [_norm(r) for r in raws]
        labs["string"].append([n.index(x) for x in n])
        labs["anchor"].append(cluster(raws, question=q)[0])
        labs["gold"].append(cluster(raws, correct=flags, question=q)[0])
    for k, l in labs.items():
        a, b = _diagnostics(facts, l)
        out[k] = {"both_correct": round(a, 4), "correct_incorrect": round(b, 4)}
    return out


if __name__ == "__main__":
    keys = sorted(os.path.basename(p)[10:-4] for p in glob.glob("outputs/hidden2x2_*.npz"))
    with Pool(13) as pool:
        rows = pool.map(run, keys)
    json.dump(rows, open("outputs/cluster_diagnostics.json", "w"), indent=2)
    for r in rows:
        print(f"{r['model']:18s} " + " | ".join(f"{k} {r[k]['both_correct']:.3f}/{r[k]['correct_incorrect']:.3f}" for k in ("string", "anchor", "gold")))
