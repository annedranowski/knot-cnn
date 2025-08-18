# knot_detector/training/loops.py
from __future__ import annotations
from typing import Callable, Tuple, List
import torch

Tensor = torch.Tensor

def _maybe_squeeze_logits(y_pred: Tensor) -> Tensor:
    # If model returns shape [B,1], make it [B]
    if y_pred.ndim == 2 and y_pred.size(-1) == 1:
        return y_pred.squeeze(1)
    return y_pred

@torch.no_grad()
def test_step(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    accuracy_fn: Callable[..., float],
    device: torch.device,
) -> Tuple[List[float], List[float], float, float]:
    """
    Runs one evaluation epoch.
    Returns: (y_pred_list, y_true_list, avg_loss, avg_acc)
    """
    model.eval()
    test_loss, test_acc = 0.0, 0.0
    y_pred_list: List[float], y_true_list: List[float] = [], []

    for X, y in data_loader:
        X, y = X.to(device), y.to(device)

        logits = model(X)
        logits = _maybe_squeeze_logits(logits)

        # Collect rounded preds to mirror the notebook behavior
        y_pred_list.extend([round(v) for v in logits.detach().cpu().tolist()])
        y_true_list.extend([round(v) for v in y.detach().cpu().tolist()])

        test_loss += loss_fn(logits, y.to(torch.float32)).item()
        test_acc  += accuracy_fn(y_true=y, y_pred=logits.round())

    n_batches = max(1, len(data_loader))
    test_loss /= n_batches
    test_acc  /= n_batches
    return y_pred_list, y_true_list, test_loss, test_acc


def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn: Callable[..., float],
    device: torch.device,
) -> Tuple[List[float], List[float], float, float]:
    """
    Runs one training epoch.
    Returns: (y_pred_list, y_true_list, avg_loss, avg_acc)
    """
    model.train()
    train_loss, train_acc = 0.0, 0.0
    y_pred_list: List[float], y_true_list: List[float] = [], []

    for X, y in data_loader:
        X, y = X.to(device), y.to(device)

        # 1) forward
        logits = model(X)
        logits = _maybe_squeeze_logits(logits)

        # Collect preds/targets like your notebook
        y_pred_list.extend([round(v) for v in logits.detach().cpu().tolist()])
        y_true_list.extend([round(v) for v in y.detach().cpu().tolist()])

        # 2) loss
        loss = loss_fn(logits, y.to(torch.float32))
        train_loss += loss.item()
        train_acc  += accuracy_fn(y_true=y, y_pred=logits.round())

        # 3–5) optimize
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    n_batches = max(1, len(data_loader))
    train_loss /= n_batches
    train_acc  /= n_batches
    return y_pred_list, y_true_list, train_loss, train_acc