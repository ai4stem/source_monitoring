"""
model.py
========
The metacognitive source-monitoring head and the composite loss, implementing:

  p_hat_j     = softmax(f_theta(u_j))                 3-class source prediction
  L_source    = -1/|L| sum_j beta_{y_j} log p_hat_{j,y_j}    weighted CE
  L_conflict  = -1/|K| sum_{j in K} log p_hat_{j,verbatim}
  L_total     = L_LM + alpha L_source + gamma L_conflict

`SourceHead` consumes the pooled claim signal u_j (from signals.py) plus, optionally,
the claim's hidden state (the "performance-ceiling" ablation).  Learnable pieces:
head_weights phi_h, pooling scorer g, and the classifier f_theta -- all tiny,
kept well under 1% of params alongside the LoRA adapters.

torch is imported lazily so the file can be inspected without it; the numpy
reference implementations of the losses ARE available everywhere for testing.
"""

from __future__ import annotations
import numpy as np

SOURCE_CLASSES = ["verbatim", "parametric", "confabulated"]
CLASS_TO_IDX = {c: i for i, c in enumerate(SOURCE_CLASSES)}


# ---------------------------------------------------------------------------
# Class weights beta_c  (effective number of samples; Cui et al. 2019)
# ---------------------------------------------------------------------------

def class_weights(counts: dict[str, int], rho: float = 0.99) -> np.ndarray:
    """beta_c = (1 - rho) / (1 - rho^{N_c}), renormalized to sum = n_classes."""
    n = np.array([max(counts.get(c, 0), 1) for c in SOURCE_CLASSES], dtype=float)
    eff = (1.0 - rho) / (1.0 - np.power(rho, n))
    eff = eff / eff.sum() * len(SOURCE_CLASSES)
    return eff


# ---------------------------------------------------------------------------
# numpy reference losses (for unit tests / analysis without torch)
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def source_loss_np(logits: np.ndarray, labels: np.ndarray,
                   beta: np.ndarray, eps: float = 1e-12) -> float:
    """logits [N,3], labels [N] in {0,1,2}, beta [3]. Weighted CE."""
    p = _softmax(logits)
    idx = np.arange(len(labels))
    ce = -np.log(p[idx, labels] + eps)
    w = beta[labels]
    return float((w * ce).mean())


def conflict_loss_np(logits: np.ndarray, conflict_mask: np.ndarray,
                     eps: float = 1e-12) -> float:
    """Push conflict claims toward 'verbatim'. logits[K_subset,3]."""
    if conflict_mask.sum() == 0:
        return 0.0
    p = _softmax(logits[conflict_mask])
    v = CLASS_TO_IDX["verbatim"]
    return float((-np.log(p[:, v] + eps)).mean())


def total_loss_np(lm_loss: float, source_logits: np.ndarray, labels: np.ndarray,
                  conflict_mask: np.ndarray, beta: np.ndarray,
                  alpha: float = 0.1, gamma: float = 0.02) -> dict:
    ls = source_loss_np(source_logits, labels, beta)
    lc = conflict_loss_np(source_logits, conflict_mask)
    total = lm_loss + alpha * ls + gamma * lc
    return {"total": total, "lm": lm_loss, "source": ls, "conflict": lc,
            "alpha": alpha, "gamma": gamma}


# ---------------------------------------------------------------------------
# torch modules (server-side)
# ---------------------------------------------------------------------------

def build_source_head(signal_dim: int = 3, hidden_dim: int | None = None,
                      use_hidden_state: bool = False, hidden_state_dim: int = 4096):
    """Returns a small torch.nn.Module mapping pooled claim features -> 3 logits.

    signal_dim: dimensionality of z (2 or 3).
    use_hidden_state: if True, concatenates the claim's mean hidden state
                      (the performance-ceiling ablation).
    """
    import torch
    import torch.nn as nn

    in_dim = signal_dim + (hidden_state_dim if use_hidden_state else 0)
    h = hidden_dim or max(8, in_dim)

    class SourceHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, h), nn.GELU(), nn.Linear(h, 3)
            )
            # learnable per-head weights phi_h are stored on the wrapper that
            # computes context_mass; the pooling scorer g is a 1-layer net:
            self.pool_scorer = nn.Sequential(nn.Linear(signal_dim, 8), nn.GELU(),
                                             nn.Linear(8, 1))

        def pool(self, z_claim):           # z_claim [T, signal_dim]
            s = self.pool_scorer(z_claim).squeeze(-1)      # [T]
            w = torch.softmax(s, dim=0)
            return (w.unsqueeze(-1) * z_claim).sum(0)      # [signal_dim]

        def forward(self, u):              # u [N, in_dim] pooled claim features
            return self.net(u)             # [N, 3] logits

    return SourceHead()


def composite_loss_torch(lm_loss, source_logits, labels, conflict_mask,
                         beta, alpha: float = 0.1, gamma: float = 0.02):
    """torch version of total_loss. Shapes mirror the numpy reference."""
    import torch
    import torch.nn.functional as F

    beta_t = torch.as_tensor(beta, dtype=source_logits.dtype,
                             device=source_logits.device)
    ls = F.cross_entropy(source_logits, labels, weight=beta_t)

    if conflict_mask.any():
        v = CLASS_TO_IDX["verbatim"]
        logp = F.log_softmax(source_logits[conflict_mask], dim=-1)
        lc = -logp[:, v].mean()
    else:
        lc = torch.zeros((), device=source_logits.device)

    total = lm_loss + alpha * ls + gamma * lc
    return total, {"lm": float(lm_loss), "source": float(ls),
                   "conflict": float(lc), "total": float(total)}
