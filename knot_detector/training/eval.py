import torch
import torch_xla

from accuracy_fn import accuracy_fn

def eval_model(model: torch.nn.Module,
               data_loader: torch_xla.distributed.parallel_loader.MpDeviceLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    y_pred_eval = []; y_target_eval = []
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            y_pred = model(X).squeeze(dim=1)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                                y_pred=y_pred.round())

            y.to('cpu'), y_pred.to('cpu')
            y_pred_eval.append(y_pred.round())
            y_target_eval.append(y)

        # Scale loss and acc
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}, y_pred_eval, y_target_eval
