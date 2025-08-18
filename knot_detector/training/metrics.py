# knot_detector/training/metrics.py
import torch
def binary_round_acc(*, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return (y_pred.round().eq(y_true).float().mean().item()) * 100.0
