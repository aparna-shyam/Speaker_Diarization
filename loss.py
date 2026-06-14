"""
loss.py
=======
Loss function for GNN speaker-embedding refinement (paper §2.4).

BCEWithLogitsLoss is used so that no separate sigmoid allocation is
needed during training — numerically more stable and memory-efficient.

Optionally, a positive-class weight can be passed to compensate for
the heavy class imbalance (most pairs are *not* the same speaker).
Typical value: weight = (N² - num_same) / num_same  ≈  #spk / 1
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Module-level singleton — re-created only when pos_weight changes
_bce: nn.BCEWithLogitsLoss | None = None
_pos_weight_val: float | None     = None


def bce_loss(
    logits:     torch.Tensor,
    gt:         torch.Tensor,
    pos_weight: float | None = None,
) -> torch.Tensor:
    """
    Binary cross-entropy (with logits) between predicted pair scores
    and the ground-truth adjacency matrix.

    Parameters
    ----------
    logits     : raw (pre-sigmoid) scores, shape (N, N)
    gt         : float32 binary matrix, shape (N, N)
    pos_weight : optional scalar weight for positive (same-speaker) pairs.
                 Helpful when speakers are few — positive pairs are rare.
                 A value of ~(N² / num_positive_pairs - 1) is a reasonable
                 starting point.  None → unweighted loss.

    Returns
    -------
    Scalar loss tensor.
    """
    global _bce, _pos_weight_val

    if pos_weight is None:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        if pos_weight != _pos_weight_val:
            pw = torch.tensor([pos_weight], dtype=torch.float,
                              device=logits.device)
            _bce           = nn.BCEWithLogitsLoss(pos_weight=pw)
            _pos_weight_val = pos_weight
        loss_fn = _bce

    return loss_fn(logits, gt)


# ---------------------------------------------------------------------------
# Convenience: compute a session-level pos_weight from the GT matrix
# ---------------------------------------------------------------------------

def compute_pos_weight(gt: torch.Tensor, eps: float = 1.0) -> float:
    """
    Returns  (# negative pairs) / (# positive pairs).

    Pass the result to bce_loss(pos_weight=...) to up-weight the rare
    same-speaker pairs.  `eps` prevents division-by-zero on edge cases.
    """
    n_pos = gt.sum().item()
    n_neg = gt.numel() - n_pos
    return float(n_neg / (n_pos + eps))
