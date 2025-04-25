import re
import sys
import torch.nn as nn
from create_model import *
import torch
from torchinfo import summary


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


def __print_model(model, model_depth):
    print('-' * 50)
    print(
        f'Summary for entrance of size (1, 35, 400, 400), (1, 11), depth {model_depth}')
    print(summary(model, [(1, 1, 35, 400, 400), (1, 11)], device='cpu',
                  dtypes=[torch.float, torch.float], depth=4))
    print('-' * 50)


def __load_resnet(args, device):
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
            state_dict = {k.replace('module.', 'model.')                          : v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
        else:
            model = torch.load(args.model_path, weights_only=False)

    freeze_bn_layers(model)
    # replace_bn_with_instancenorm(model)
    model = model.to(device)

    __print_model(model, model_depth)

    return model, model_depth


def __load_densenet(args, device):
    match = re.search(r"densenet_(\d+)", args.model_path)
    if match:
        model_depth = int(match.group(1))
    else:
        print("Model depth not found, please the format: densenet_dethp*",
              file=sys.stderr)

        return None, None

    if args.from_scratch:

        model = DenseNet(
            model_depth,
            args.fc_layers_arch_densenet,
            dropout=args.use_dropout_densenet,
            batch_norm=args.use_bn_densenet,
            use_metadata=args.use_metadata_densenet
        ).to(device)

        __print_model(model, model_depth)

        return model, model_depth

    model = DenseNet(
        model_depth,
        args.fc_layers_arch_densenet,
        dropout=args.use_dropout_densenet,
        batch_norm=args.use_bn_densenet,
        use_metadata=args.use_metadata_densenet
    )
    model.load_state_dict(torch.load(args.model_path))

    model = model.to(device)

    return model, model_depth


def load_pretrained_model(args, device):
    if 'resnet' in args.model_path:
        return __load_resnet(args, device)
    elif 'densenet' in args.model_path:
        return __load_densenet(args, device)
    else:
        raise ValueError('model architecture unknown')
