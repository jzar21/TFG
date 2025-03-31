import re
import sys
import torch.nn as nn
from create_model import *
import torch
from torchsummary import summary


def freeze_bn_layers(model):
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm3d) or isinstance(child, nn.BatchNorm1d):
            # child.requires_grad_(False)
            # child.track_running_stats = False
            child.running_mean = None
            child.running_var = None
        else:
            freeze_bn_layers(child)


def replace_bn_with_instancenorm(model):
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm3d):
            new_layer = nn.InstanceNorm3d(child.num_features)
            setattr(model, name, new_layer)
        elif isinstance(child, nn.BatchNorm1d):
            new_layer = nn.InstanceNorm1d(child.num_features)
            setattr(model, name, new_layer)
        else:
            replace_bn_with_instancenorm(child)


def load_pretrained_model(args, device):
    match = re.search(r"resnet_(\d+)", args.model_path)

    if match:
        model_depth = int(match.group(1))
    else:
        print("Model depth not found, please the format: resnet_dethp*",
              file=sys.stderr)

        return None

    model = ResNet3D(model_depth, fc_layers=args.fc_layers_arch,
                     dropout=args.dropout).to(device)
    if args.classification:
        model = ResNet3DBinaryClasificacion(model_depth, fc_layers=args.fc_layers_arch,
                                            dropout=args.dropout).to(device)

    print(f'Depth: {model_depth}')
    if not args.from_scratch:
        if args.pretrain_med_net:
            checkpoint = torch.load(args.model_path)
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('module.', 'model.'): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
        else:
            model = torch.load(args.model_path, weights_only=False)

    freeze_bn_layers(model)
    # replace_bn_with_instancenorm(model)
    model = model.to(device)

    # print('-' * 50)
    # print(
    #     f'Summary for entrance of size (1, 20, 100, 100), depth {model_depth}')
    # print(summary(model, (1, 1, 20, 100, 100)))
    # print('-' * 50)

    return model, model_depth
