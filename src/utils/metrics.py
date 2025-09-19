from __future__ import annotations

from typing import Dict

import torch


@torch.no_grad()
def binary_metrics(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None, threshold: float = 0.5) -> Dict[str, float]:
    """Compute simple binary metrics from logits (B,1) vs targets (B,1).

    If mask provided, it should be broadcastable to logits shape to ignore invalid entries.
    """
    probs = torch.sigmoid(logits)
    if mask is not None:
        probs = probs[mask]
        targets = targets[mask]
    preds = (probs >= threshold).float()
    tp = (preds.eq(1) & targets.eq(1)).sum().item()
    tn = (preds.eq(0) & targets.eq(0)).sum().item()
    fp = (preds.eq(1) & targets.eq(0)).sum().item()
    fn = (preds.eq(0) & targets.eq(1)).sum().item()
    eps = 1e-9
    prec = tp / max(eps, (tp + fp))
    rec = tp / max(eps, (tp + fn))
    f1 = 2 * prec * rec / max(eps, (prec + rec))
    acc = (tp + tn) / max(eps, (tp + tn + fp + fn))
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1), "accuracy": float(acc)}
