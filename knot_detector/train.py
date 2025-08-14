import torch
import tqdm

# Training function(s) are model-agnostic (accept a model as argument)

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    y_pred_train, y_target_train = [], []
    for (X, y) in data_loader:
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X).squeeze(dim=1)

        for i in y_pred.tolist():
          y_pred_train.append(round(i))
        for i in y.tolist():
          y_target_train.append(round(i))

        # 2. Calculate loss
        loss = loss_fn(y_pred, y.type(torch.float32))
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.round()) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"\nTrain loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
    return y_pred_train, y_target_train, train_loss.cpu().detach().numpy(), train_acc

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              threshold: float = 0.001,
              device: torch.device = device,
              scheduler: torch.optim.lr_scheduler = None,
              save_path: str = None):
    test_loss, test_acc = 0, 0
    y_pred_test, y_target_test = [], []
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X).squeeze(dim=1)

            for i in test_pred.tolist():
              y_pred_test.append(round(i))
            for i in y.tolist():
              y_target_test.append(round(i))

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y.type(torch.float32))
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.round() # Go from logits -> pred labels
            )

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

        if scheduler != None:
          scheduler.step(test_loss)

        if test_acc > best_acc:
          best_fold = fold
          if save_path != None:
            torch.save(model.state_dict(), save_path + f'{model.__class__.__name__}_best.pth')

        print(f"\nTest loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
        return y_pred_test, y_target_test, test_loss.cpu().detach().numpy(), test_acc

def train_loop(model, data_loader, loss_fn, optimizer, accuracy_fn, num_epochs, scheduler=None):
    for epoch in tqdm.notebook.tqdm(range(num_epochs)):
        print(f'\nEpoch: {epoch+1}')
        if scheduler != None:
            print(f'LR: {scheduler.get_last_lr()}')
        
        y_pred_train, y_target_train, train_loss, train_acc = train_step(data_loader=train_dataloader,
                                                                         model=model_2,
                                                                         loss_fn=loss_fn,
                                                                         optimizer=optimizer,
                                                                         accuracy_fn=accuracy_fn,
                                                                        )
        train_losses.append(train_loss); train_accuracies.append(train_acc)
        
        y_pred_valid, y_target_valid, valid_loss, valid_acc = test_step(data_loader=valid_dataloader,
                                                                        model=model_2,
                                                                        loss_fn=loss_fn,
                                                                        accuracy_fn=accuracy_fn,
                                                                        scheduler=scheduler
                                                                        )
        valid_losses.append(valid_loss); valid_accuracies.append(valid_acc)
        
        torch.cuda.empty_cache()
