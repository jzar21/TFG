from resnets_3d.models.resnet import generate_model
import torch.nn as nn


class ResNet3D_Regresion(nn.Module):
    def __init__(self, model_depth: int = 18, n_input_channels: int = 1,
                 n_classes: int = 1):
        self.model = generate_model(
            model_depth, n_input_channels=n_input_channels, n_classes=n_classes)

    def forward(self, x):
        return self.model(x)
