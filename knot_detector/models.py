import torch.nn as nn

from torch import nn

# Create a model with non-linear and linear layers
class KnotsModelVanila(nn.Module):
    def __init__(self, input_shape: int, hidden_units_1: int, hidden_units_2: int, hidden_units_3:int, hidden_units_4: int, output_shape: int): # try to change the number of hidden layers
        super().__init__()
        self.dropout = nn.Dropout(p=0.9)

        self.layer_prepare = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(input_shape)
        )
        self.layer_1 = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=hidden_units_1),
            nn.ELU(),
            nn.BatchNorm1d(hidden_units_1)
        )
        self.layer_2 = nn.Sequential(
            nn.Linear(in_features=hidden_units_1, out_features=hidden_units_2),
            nn.ELU(),
            nn.BatchNorm1d(hidden_units_2)
        )
        self.layer_3 = nn.Sequential(
            nn.Linear(in_features=hidden_units_2, out_features=hidden_units_3),
            nn.ELU(),
            nn.BatchNorm1d(hidden_units_3)
        )
        self.layer_4 = nn.Sequential(
            nn.Linear(in_features=hidden_units_3, out_features=hidden_units_4),
            nn.ELU()
        )
        self.layer_5 = nn.Sequential(
            nn.Linear(in_features=hidden_units_4, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
      x = self.layer_prepare(x)
      #x = self.dropout(x)
      x = self.layer_1(x)
      #x = self.dropout(x)
      x = self.layer_2(x)
      #x = self.dropout(x)
      x = self.layer_3(x)
      #x = self.dropout(x)
      x = self.layer_4(x)
      x = self.dropout(x)
      x = self.layer_5(x)
      return x

# Create a convolutional neural network
class KnotsModelCNN(nn.Module):
    """
    Model architecture copying TinyVGG from:
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, output_shape: int):
        super().__init__()

        self.conv_1 = nn.Sequential(
          nn.Conv2d(1, 4, kernel_size=11, stride=1, dilation=2, padding=0),
          nn.BatchNorm2d(4),
          nn.ReLU(),

          nn.MaxPool2d(kernel_size=2),

          nn.Conv2d(4, 16, kernel_size=5, stride=1, dilation=2, padding=0),
          nn.BatchNorm2d(16),
          nn.ReLU(),

          nn.MaxPool2d(kernel_size=2),

          nn.Conv2d(16, 64, kernel_size=3, stride=1, dilation=2, padding=0),
          nn.BatchNorm2d(64),
          nn.ReLU(),

          nn.MaxPool2d(kernel_size=2),

          nn.Conv2d(64, 256, kernel_size=3, stride=1, dilation=2, padding=0),
          nn.BatchNorm2d(256),
          nn.ReLU(),

          nn.MaxPool2d(kernel_size=2),

          nn.Conv2d(256, 361, kernel_size=3, stride=1, padding=0),
          nn.BatchNorm2d(361),
          nn.ReLU(),

          nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(361*(12)**2),

            nn.Dropout(p=0.7),

            nn.Linear(in_features=361*(12)**2,
                      out_features=4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),

            nn.Dropout(p=0.7),

            nn.Linear(in_features=4096,
                      out_features=4096),
            nn.ReLU(),

            nn.Linear(4096, output_shape)
        )

    def forward(self, x: torch.Tensor):
      x = self.conv_1(x)
      #print(x.shape)
      x = self.classifier(x)
      return x
