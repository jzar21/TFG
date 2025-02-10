from resnets_3d.models.resnet import generate_model
import torch.nn as nn


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
            layers.append(nn.BatchNorm1d(fc_layers[i]))

        layers.append(nn.Linear(fc_layers[-2], fc_layers[-1]))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x


class ResNet3D_Clasificacion(nn.Module):
    def __init__(self, model_depth: int = 18, n_input_channels: int = 1, min_age=14,
                 max_age=25, fc_layers: list = [1024, 512, 256, 1]):
        super(ResNet3D_Regresion, self).__init__()
        self.model = generate_model(
            model_depth, n_input_channels=n_input_channels, n_classes=fc_layers[0])

        fc_layers[-1] = max_age - min_age + 1

        layers = [nn.ReLU(), nn.BatchNorm1d(fc_layers[0])]

        for i in range(1, len(fc_layers) - 1):
            layers.append(
                nn.Linear(fc_layers[i - 1], fc_layers[i])
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(fc_layers[i]))

        layers.append(nn.Linear(fc_layers[-2], fc_layers[-1]))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x
