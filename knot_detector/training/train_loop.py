# Functions used to train a model

import gc

import time
from accuracy_fn import accuracy_fn

import tqdm
from tqdm.notebook import tqdm as tqdmn

import torch
import torch_xla
import torch_xla.core.xla_model as xm

# Basic functions which perform one step
def train_step(model: torch.nn.Module,
               epoch: int,
               start_epoch: int,
               data_loader: torch_xla.distributed.parallel_loader.MpDeviceLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               scheduler: torch.optim.lr_scheduler = None):
    train_loss, train_acc = 0, 0
    y_pred_train, y_target_train = [], []
    times_epoch = []

    optimizer.zero_grad()
    for i, (X, y) in enumerate(tqdmn(data_loader)):
        start = time.time()

        y_pred = model(X).squeeze(dim=1)

        loss = loss_fn(y_pred, y.type(torch.float32))
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.round())

        loss.backward()

        if (i+1) % (BATCH_SIZE//8) == 0:
          xm.optimizer_step(optimizer)
          optimizer.zero_grad()

        times_epoch.append(time.time()-start)

        xm.mark_step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print("\nTrain loss: {:.5f} | Train accuracy: {:.2f}%".format(train_loss, train_acc))

    if scheduler != None and epoch >= start_epoch:
      try:
        scheduler.step(train_loss)
      except:
        scheduler.step()
    lrs.append(optimizer.param_groups[0]['lr'])

    del y_pred, loss
    return y_pred_train, y_target_train, train_loss.detach().cpu().numpy(), train_acc, times_epoch

def test_step(model: torch.nn.Module,
              data_loader: torch_xla.distributed.parallel_loader.MpDeviceLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              scheduler: torch.optim.lr_scheduler = None,
              threshold: float = 0.001):
    test_loss, test_acc = 0, 0
    y_pred_test, y_target_test = [], []
    with torch.no_grad():
      model.eval()
      for X, y in data_loader:
          gc.collect()
        
          test_pred = model(X).squeeze(dim=1)

          test_loss += loss_fn(test_pred, y.type(torch.float32))
          test_acc += accuracy_fn(y_true=y,
              y_pred=test_pred.round() # Go from logits -> pred labels
          )

          xm.mark_step()

      test_loss /= len(data_loader)
      test_acc /= len(data_loader)

      print("\nTest loss: {:.5f} | Test accuracy: {:.2f}%\n".format(test_loss, test_acc))

      return test_pred, y_target_test, test_loss.cpu().detach().numpy(), test_acc

# Different statistics; used to plot time usage/learning rate's changes/confusion matrices etc.
y_pred_train, y_target_train, train_losses, train_accuracies = torch.Tensor(), torch.Tensor(), [], []
y_pred_test, y_target_test, test_losses, test_accuracies = torch.Tensor(), torch.Tensor(), [], []
times = []

# Main function
def train_fn(index: int,
            model: torch.nn.Module,
            start_epoch: int,
            train_data_loader: torch_xla.distributed.parallel_loader.MpDeviceLoader,
            valid_dataloader: torch_xla.distributed.parallel_loader.MpDeviceLoader,
            loss_fn: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            accuracy_fn,
            num_epochs: int = None,
            scheduler: torch.optim.lr_scheduler = None):
  for epoch in tqdmn(range(num_epochs)):
    gc.collect()
    print(f"Epoch {epoch+1}/{num_epochs}")

    start = time.time()
    y_pred_train, y_target_train, train_loss, train_acc, times_epoch = train_step(
               model,
               epoch,
               start_epoch,
               train_data_loader,
               loss_fn,
               optimizer,
               accuracy_fn,
               scheduler)
    end = time.time()
    print(f'Train {epoch+1} epoch : {end-start} s')
    train_losses.append(train_loss); train_accuracies.append(train_acc)
    times.append(times_epoch)

    print('----------')

  start = time.time()
  y_pred_test, y_target_test, test_loss, test_acc = test_step(
              model,
              valid_data_loader,
              loss_fn,
              accuracy_fn,
              scheduler=scheduler)
  end = time.time()
  print(f'Test {epoch+1} epoch : {end-start} s')
  test_losses.append(test_loss); test_accuracies.append(test_acc)

# Function used to perform main function on TPUs
def train_loop(model: torch.nn.Module,
               start_epoch: int,
               train_data_loader: torch_xla.distributed.parallel_loader.MpDeviceLoader,
               valid_dataloader: torch_xla.distributed.parallel_loader.MpDeviceLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               num_epochs: int = None,
               scheduler: torch.optim.lr_scheduler = None):
  os.environ["XLA_TPU_DEVICES"] = ",".join([f"tpu_device_{i}" for i in range(8)])
  xmp.spawn(
        train_fn,
        args=(model, start_epoch, train_data_loader, valid_dataloader, loss_fn, optimizer, accuracy_fn, num_epochs, scheduler),
        nprocs=1,
        start_method='fork'
    )
