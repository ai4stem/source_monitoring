"""
env_load.py
===========
Tiny zero-dependency .env loader. Importing this module (or calling load_env())
reads a .env file from the project root / cwd and populates os.environ for any
key not already set. Avoids needing python-dotenv.

.env format (one per line, # for comments):
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...
    HF_TOKEN=hf_...

Search order: ./.env, ../.env, then the dir of this file's parent. First found
wins. Existing environment variables are NOT overwritten (so an exported var
still takes precedence over the file).
"""

from __future__ import annotations
import os


def _candidate_paths() -> list[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.getcwd(), "..", ".env"),
        os.path.join(here, "..", ".env"),     # project root (src/..)
        os.path.join(here, ".env"),
    ]


def load_env(verbose: bool = False) -> str | None:
    """Load the first .env found. Returns the path used, or None."""
    for path in _candidate_paths():
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            continue
        loaded = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
                    loaded += 1
        if verbose:
            print(f"[env] loaded {loaded} vars from {path}")
        return path
    if verbose:
        print("[env] no .env found (using existing environment only)")
    return None


# auto-load on import so any module that does `import env_load` gets keys set
load_env()


if __name__ == "__main__":
    p = load_env(verbose=True)
    # show which expected keys are present (masked)
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN"):
        v = os.environ.get(k)
        shown = (v[:6] + "..." + v[-4:]) if v and len(v) > 12 else ("set" if v else "MISSING")
        print(f"  {k}: {shown}")
