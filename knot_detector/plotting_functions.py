import torch
from torchvision import transforms

def plot_weights(layers: list):
  fig, axs = plt.subplots(ncols=1, nrows=len(layers), figsize=(7, 7*len(layers)))
  for i, layer in enumerate(layers):
    if type(layer) == torch.nn.modules.linear.Linear:
      weight = torch.sum(layer.weight, dim=0)
      siz = int(weight.shape[0]**(1/2))
      weight = torch.unflatten(weight, dim=0, sizes=(siz, siz))
    elif type(layer) == torch.nn.modules.conv.Conv2d:
      if 1 not in layer.kernel_size:
        weight = torch.sum(torch.sum(layer.weight, dim=0), dim=0)
    axs[i].imshow(weight.cpu().detach().numpy())

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

def plot_weights_image_cvt(model, img):
  fig, axs = plt.subplots(ncols=(len(model.stages)+1)//2, nrows=2, figsize=((len(model.stages)+1)//2*7, 2*7))
  axs[0][0].axis('off')
  axs[0][0].text(0.5, -0.01, "Original image", ha='center', va='top', transform=axs[0][0].transAxes, fontsize=20)
  axs[0][0].imshow(torch.sum(img, dim=1).permute(1, 2, 0).detach().numpy())
  for i, layer in enumerate(model.stages):
    img = layer(img)
    img_show = torch.sum(img, dim=1)
    axs[(i+1)//2][(i+1)%2].axis('off')
    axs[(i+1)//2][(i+1)%2].text(0.5, -0.01, f"After {i+1} stage", ha='center', va='top', transform=axs[(i+1)//2][(i+1)%2].transAxes, fontsize=20)
    axs[(i+1)//2][(i+1)%2].imshow(img_show.permute(1, 2, 0).squeeze(dim=0).detach().numpy())
