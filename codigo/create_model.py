import monai.networks
import monai.networks.nets
from resnets_3d.models.resnet import generate_model
import torch.nn as nn
import monai


class ResNet3D_Regresion(nn.Module):
    def __init__(self, model_depth: int = 18, n_input_channels: int = 1,
                 fc_layers: list = [1024, 512, 256, 1]):
        super(ResNet3D_Regresion, self).__init__()
        self.model = generate_model(
            model_depth, n_input_channels=n_input_channels, n_classes=fc_layers[0])

        layers = [nn.ReLU(), nn.BatchNorm1d(fc_layers[0])]

        for i in range(1, len(fc_layers) - 1):
            layers.append(
                nn.Linear(fc_layers[i - 1], fc_layers[i])
            )
            layers.append(nn.ReLU())
            # layers.append(nn.BatchNorm1d(fc_layers[i]))

        layers.append(nn.Linear(fc_layers[-2], fc_layers[-1]))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x


class ResNet3D101_MONAI(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet3D101_MONAI, self).__init__()

        self.model = monai.networks.nets.resnet101(
            spatial_dims=3, n_input_channels=1,
            num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
