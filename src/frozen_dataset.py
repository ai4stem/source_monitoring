"""
frozen_dataset.py
=================
Guarantees that every run -- across every model, every day -- uses the EXACT
same dataset. The dataset is built once, saved, and content-hashed; subsequent
runs load the saved file and verify the hash. If you rebuild, the hash changes
and the manifest records it, so a comparison can never silently drift.

Why this matters: comparing models (8B vs 30B vs GPT-4o) only means something
if they all saw identical items. This module makes that a hard guarantee.

Usage:
    python frozen_dataset.py freeze   # build once, write dataset + manifest
    python frozen_dataset.py verify   # check the saved dataset matches manifest
    # in code:
    insts = load_frozen("data/frozen/dataset.jsonl")   # raises if hash mismatch
"""

from __future__ import annotations
import hashlib
import json
import os
from schema import load_dataset, save_dataset
from build_dataset import build_all

MANIFEST = "data/frozen/manifest.json"
FROZEN_PATH = "data/frozen/dataset.jsonl"


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze(seed: int = 0, path: str = FROZEN_PATH, manifest: str | None = None) -> dict:
    """Build the dataset deterministically and freeze it with a content hash."""
    if manifest is None:
        manifest = os.path.join(os.path.dirname(path), "manifest.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    instances = build_all(seed=seed)
    save_dataset(instances, path)            # deterministic given seed
    digest = _file_hash(path)
    info = {
        "path": path,
        "seed": seed,
        "sha256": digest,
        "n_instances": len(instances),
        "builder_version": "1.0",
    }
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"[freeze] {len(instances)} instances, sha256={digest[:12]}...")
    print(f"[freeze] dataset -> {path}\n[freeze] manifest -> {manifest}")
    return info


def verify(path: str = FROZEN_PATH, manifest: str | None = None) -> bool:
    """Return True iff the saved dataset still matches the frozen manifest hash."""
    if manifest is None:
        manifest = os.path.join(os.path.dirname(path), "manifest.json")
    if not (os.path.exists(path) and os.path.exists(manifest)):
        raise FileNotFoundError("frozen dataset or manifest missing; run `freeze` first")
    with open(manifest, encoding="utf-8") as f:
        info = json.load(f)
    actual = _file_hash(path)
    ok = (actual == info["sha256"])
    status = "OK" if ok else "MISMATCH"
    print(f"[verify] {status}: expected {info['sha256'][:12]}..., got {actual[:12]}...")
    return ok


def load_frozen(path: str = FROZEN_PATH, manifest: str | None = None):
    """Load the frozen dataset, refusing to proceed if the hash drifted."""
    if not verify(path, manifest):
        raise RuntimeError(
            "Frozen dataset hash mismatch -- the dataset changed since it was "
            "frozen. Comparisons would be invalid. Re-freeze intentionally if "
            "this is expected (and re-run ALL models)."
        )
    return load_dataset(path)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["freeze", "verify"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--path", default=FROZEN_PATH)
    args = p.parse_args()
    if args.cmd == "freeze":
        freeze(seed=args.seed, path=args.path)
    else:
        verify(path=args.path)
