"""
schema.py
=========
Data schema for the Source Monitoring study.

Implements the record structure we designed:
  - Three item types: drm | fact | conflict
  - Seven control probe kinds (critical_lure, studied, unrelated, parametric_fact,
    confabulated_fact, conflict, non_conflict_control)
  - Three source labels: verbatim | parametric | confabulated
  - conflict flag (kappa) and a `labeled` flag (gray-zone items excluded from loss)

No torch / transformers dependency here -- pure dataclasses + JSON I/O so the
schema can be built, saved, loaded, and validated on any machine.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
import json
import os

# "pending" = fact item whose final label is assigned later by model-relative
# stability probing. It is labeled=True (we intend to use it) but not yet resolved.
SourceLabel = Literal["verbatim", "parametric", "confabulated", "pending"]
ItemType = Literal["drm", "fact", "conflict"]
ProbeKind = Literal[
    "critical_lure",        # DRM lure -> confabulated if model says "yes"
    "studied",              # DRM studied item (positive control) -> verbatim
    "unrelated",            # DRM unrelated lure (negative control) -> no label
    "parametric_fact",      # closed-book fact the model stably knows -> parametric
    "confabulated_fact",    # closed-book unanswerable -> confabulated
    "conflict",             # context contradicts parametric -> verbatim (kappa=1)
    "non_conflict_control", # context agrees/irrelevant -> parametric (over-deference control)
]


@dataclass
class Claim:
    """A single claim span within an instance.

    span is stored as a [start_char, end_char] pair on the *target* text.
    Token spans are re-projected per-model at tokenization time (model-relative).
    """
    text: str
    source: Optional[SourceLabel] = None   # None for unlabeled / negative-control probes
    conflict: int = 0                       # kappa_j in {0,1}
    labeled: bool = True                    # False -> excluded from the loss (gray zone)
    char_span: Optional[list[int]] = None   # [start, end] on target text, optional

    def validate(self) -> None:
        if self.labeled:
            assert self.source in ("verbatim", "parametric", "confabulated", "pending"), (
                f"labeled claim must have a valid source, got {self.source!r}"
            )
        assert self.conflict in (0, 1), f"conflict must be 0/1, got {self.conflict}"

    def is_resolved(self) -> bool:
        """True if this claim has a final (non-pending) label usable in the loss."""
        return self.labeled and self.source in ("verbatim", "parametric", "confabulated")


@dataclass
class Instance:
    """One training/evaluation instance."""
    id: str
    item_type: ItemType
    probe_kind: ProbeKind
    context: str                 # the verbatim evidence text shown to the model ("" for closed-book)
    question: str
    target: str                  # the correct answer text
    claims: list[Claim] = field(default_factory=list)

    # optional metadata used by specific item types
    lure: Optional[str] = None              # DRM critical lure word
    list_type: Optional[str] = None         # e.g. "high_BAS" / "low_BAS"
    parametric_lure: Optional[str] = None   # conflict: the parametric temptation
    stability: Optional[dict] = None        # fact: probing result record
    answer_aliases: Optional[list] = None   # fact: acceptable answer variants
    context_char_span: Optional[list[int]] = None  # span of evidence inside context

    def validate(self) -> None:
        assert self.item_type in ("drm", "fact", "conflict")
        for c in self.claims:
            c.validate()
        # conflict claims must be verbatim by construction (once resolved)
        for c in self.claims:
            if c.conflict == 1 and c.is_resolved():
                assert c.source == "verbatim", (
                    f"conflict claim {self.id} must be verbatim, got {c.source}"
                )

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Instance":
        claims = [Claim(**c) for c in d.pop("claims", [])]
        return cls(claims=claims, **d)


# ---------------------------------------------------------------------------
# Dataset I/O  (JSONL: one instance per line)
# ---------------------------------------------------------------------------

def save_dataset(instances: list[Instance], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for inst in instances:
        inst.validate()
    with open(path, "w", encoding="utf-8") as f:
        for inst in instances:
            f.write(json.dumps(inst.to_dict(), ensure_ascii=False) + "\n")


def load_dataset(path: str) -> list[Instance]:
    instances: list[Instance] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instances.append(Instance.from_dict(json.loads(line)))
    return instances


def dataset_summary(instances: list[Instance]) -> dict:
    """Quick label/type distribution -- feeds the class-weight beta_c later."""
    from collections import Counter
    by_type = Counter(i.item_type for i in instances)
    by_probe = Counter(i.probe_kind for i in instances)
    by_source = Counter(
        c.source for i in instances for c in i.claims if c.is_resolved()
    )
    n_labeled = sum(1 for i in instances for c in i.claims if c.is_resolved())
    n_pending = sum(1 for i in instances for c in i.claims
                    if c.labeled and c.source == "pending")
    n_unlabeled = sum(1 for i in instances for c in i.claims if not c.labeled)
    return {
        "n_instances": len(instances),
        "by_item_type": dict(by_type),
        "by_probe_kind": dict(by_probe),
        "by_source_label": dict(by_source),
        "n_resolved_claims": n_labeled,
        "n_pending_claims": n_pending,
        "n_unlabeled_claims": n_unlabeled,
    }
