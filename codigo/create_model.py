import torch
import torch.nn as nn
from MedicalNet.models.resnet import ResNet, Bottleneck


class ResNet3D_Regresion(ResNet):
    def __init__(self,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_output,
                 shortcut_type='B',
                 no_cuda=False):
        self.inplanes = sample_input_D
        self.no_cuda = no_cuda
        super(ResNet3D_Regresion, self).__init__(Bottleneck, layers, sample_input_D,
                                                 sample_input_H,
                                                 sample_input_W,
                                                 num_seg_classes=1,
                                                 shortcut_type=shortcut_type,
                                                 no_cuda=no_cuda)

        self.poll = nn.AdaptiveAvgPool3d(
            (1, self.inplanes, Bottleneck.expansion))
        self.fc = nn.Linear(self.inplanes * Bottleneck.expansion, num_output)

    def forward(self, x):
        x = super(ResNet3D_Regresion, self).forward(x)
        x = self.poll(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
