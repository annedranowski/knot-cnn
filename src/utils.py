import torch

def accuracy_fn(y_true, y_pred): # from https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# Function to count model complexity
def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Functions for plotting weights

# Function for plotting only weights
def plot_weights(layers: list): 
  fig, axs = plt.subplots(ncols=1, nrows=len(layers), figsize=(7, 7*len(layers)))
  for i, layer in enumerate(layers):
    if type(layer) == torch.nn.modules.linear.Linear:
      weight = torch.sum(layer.weight, dim=0)
      siz = int(weight.shape[0]**(1/2))
      weight = torch.unflatten(weight, dim=0, sizes=(siz, siz))
    elif type(layer) == torch.nn.modules.conv.Conv2d:
      weight = torch.sum(torch.sum(layer.weight, dim=0), dim=0)
    axs[i].imshow(weight.cpu().detach().numpy())
      
# Function for plotting weights and intermediate results
def plot_weights_image(layers: list, func_length: int, img: torch.Tensor): 
  fig, axs = plt.subplots(ncols=2, nrows=func_length, figsize=(2*7, 7*func_length))
  i = -1
  for layer in layers:
    img = img.cpu()
    #print(type(layer), img.shape)
    if type(layer) == torch.nn.modules.linear.Linear:
      i += 1
      # how much does image activate each neuron
      img = img.squeeze()
      if torch.Tensor.dim(img) == 1:
        siz = int(img.shape[0]**(1/2))
        img = torch.unflatten(img, dim=0, sizes=(siz, siz))

      weight = torch.sum(layer.weight, dim=0).cpu()
      siz = int(weight.shape[0]**(1/2))
      weight = torch.unflatten(weight, dim=0, sizes=(siz, siz))
      mult = torch.mul(weight.cpu(), img.cpu()).cpu()
      axs[i][0].imshow(mult.detach().numpy())

      # output
      img = torch.flatten(img).unsqueeze(dim=0)
      img = layer(img.cuda()).cpu()

      siz = int(img.shape[1]**(1/2))
      img = torch.unflatten(img, dim=1, sizes=(siz, siz))

      axs[i][1].imshow(img.squeeze(dim=0).detach().numpy())

    elif type(layer) == torch.nn.modules.conv.Conv2d:
      i += 1
      weight = torch.sum(torch.sum(layer.weight, dim=0), dim=0)
      axs[i][0].imshow(weight.cpu().detach().numpy())

      img = layer(img.cuda()).cpu()

      img_show = torch.sum(img, dim=0)

      axs[i][1].imshow(img_show.squeeze(dim=0).detach().numpy())
    elif type(layer) == torch.nn.modules.flatten.Flatten:
      img = layer(img.unsqueeze(dim=0).cuda()).squeeze().cpu()
    elif type(layer) == torch.nn.modules.batchnorm.BatchNorm1d or type(layer) == torch.nn.modules.batchnorm.BatchNorm2d:
      pass
    else:
      img = layer(img.cuda()).cpu()
