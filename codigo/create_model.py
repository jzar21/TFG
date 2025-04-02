from resnets_3d.models.resnet import generate_model
import torch.nn as nn


class ResNet3D(nn.Module):
    def __init__(self, model_depth: int = 18, n_input_channels: int = 1,
                 fc_layers: list = [1024, 512, 256, 1], dropout: bool = False):
        super(ResNet3D, self).__init__()
        self.model = generate_model(
            model_depth, n_input_channels=n_input_channels, n_classes=fc_layers[0])

        layers = [nn.Linear(self.model.fc.in_features, fc_layers[0])]

        if len(fc_layers) > 1:
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(p=0.5))

            for i in range(1, len(fc_layers) - 1):
                layers.append(
                    nn.Linear(fc_layers[i - 1], fc_layers[i])
                )
                layers.append(nn.ReLU())
                # layers.append(nn.BatchNorm1d(fc_layers[i]))
                if dropout:
                    layers.append(nn.Dropout(p=0.5))

            layers.append(nn.Linear(fc_layers[-2], fc_layers[-1]))

        self.fc = nn.Sequential(*layers)
        self.metadata_mlp = nn.Sequential(
            nn.Linear(11, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.model.fc.in_features),
            nn.Sigmoid(),
        )
        self.model.fc = nn.Identity()

    def forward(self, img, metadata):
        img = self.model(img)
        metadata = self.metadata_mlp(metadata)

        return self.fc(img * metadata)


class ResNet3DBinaryClasificacion(ResNet3D):
    def __init__(self, model_depth=18, n_input_channels=1, fc_layers=[1024, 512, 256, 1],
                 dropout: bool = False):
        fc_layers[-1] = 1
        super().__init__(model_depth, n_input_channels, fc_layers, dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = super().forward(x)
        x = self.sigmoid(x)
        return x
