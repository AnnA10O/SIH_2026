"""
Binary Focal Loss for extreme class imbalance.

Reference: Lin et al. (2017) — Focal Loss for Dense Object Detection (RetinaNet).
Adapted for scalar logits (binary classification) from the original multi-class form.

Usage in train_neural_nowcaster_v2.py:
    from src.losses.focal_loss import FocalLoss
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")
    loss = criterion(logits, targets)           # per-sample loss
    weighted_loss = (loss * sample_weights).mean()
    weighted_loss.backward()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss — Lin et al. (2017), adapted for scalar logits.

    Compared to BCEWithLogitsLoss(pos_weight=W):
      - pos_weight applies the same W× penalty to every positive example.
      - FocalLoss instead scales loss per-example by (1 - p_t)^gamma, where
        p_t is the model's confidence on the correct class.
      - Easy examples (high p_t) receive a suppressed gradient; hard examples
        (low p_t) receive a near-full gradient. This concentrates training on
        the ambiguous rainstorms driving FAR up, not the already-correct ones.

    Args:
        alpha (float):
            Positive-class prior weight in [0, 1].
            alpha_t = alpha for positives, (1 - alpha) for negatives.
            Sweep: {0.25, 0.50, 0.75}. Higher values up-weight the rare class.
        gamma (float):
            Focusing exponent >= 0.
            gamma=0 recovers standard binary cross-entropy (no focusing).
            Sweep: {2, 3, 4}. Higher gamma = stronger suppression of easy examples.
        reduction (str):
            'none'  — return per-sample loss tensor (required for sample weighting).
            'mean'  — return scalar mean loss.
            'sum'   — return scalar sum loss.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "none"):
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction!r}")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  Raw (pre-sigmoid) model output, shape (N,) or (N, 1).
            targets: Binary labels {0.0, 1.0}, same shape as logits.
        Returns:
            Loss tensor. Shape (N,) if reduction='none', scalar otherwise.
        """
        logits = logits.view(-1)
        targets = targets.view(-1)

        # Standard binary cross-entropy (numerically stable, per-sample)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Model confidence on the correct class
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)

        # Alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        # Focal modulation: down-weight easy examples
        loss = alpha_t * (1.0 - p_t) ** self.gamma * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # 'none' — caller applies sample weights externally
