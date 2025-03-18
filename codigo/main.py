from train import train
import torch
import torch.nn as nn
import torch.optim as optim
from data_loaders import *
from torch.utils.data import DataLoader
from datetime import datetime
import torchvision
from create_model import *
from load_model import *
import numpy as np
import monai
from plots import *
from program_args import *
import json


def get_loss_fun(args):
    if args.loss == 'MSE':
        return nn.MSELoss()
    if args.loss == 'BCE':
        return nn.BCELoss()


def get_optimizer(model, args):
    if args.optimizer == 'adamw':
        return optim.AdamW(model.parameters(), lr=args.learning_rate)


def create_ds(args, transform_train, transform):

    if args.regresion:
        train_ds = DataSetMRIs(
            args.train_folder, transform=transform_train, num_central_images=args.num_slices)
        valid_ds = DataSetMRIs(
            args.valid_folder, transform=transform, num_central_images=args.num_slices)
        test_ds = DataSetMRIs(args.test_folder, transform=transform,
                              num_central_images=args.num_slices)
    else:
        train_ds = DataSetMRIClassification(
            args.train_folder, transform=transform_train, num_central_images=args.num_slices)
        valid_ds = DataSetMRIClassification(
            args.valid_folder, transform=transform, num_central_images=args.num_slices)
        test_ds = DataSetMRIClassification(args.test_folder, transform=transform,
                                           num_central_images=args.num_slices)

    return train_ds, valid_ds, test_ds


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    if args.train:
        model, model_depth = load_pretrained_model(args, device)

        model.to(device)

        transform_train = monai.transforms.Compose([
            torchvision.transforms.Resize((400, 400)),
            torchvision.transforms.RandomHorizontalFlip(p=0.25),
            torchvision.transforms.RandomCrop((400, 400)),
            torchvision.transforms.RandomPerspective(p=0.25),
            monai.transforms.RandRotate(
                range_x=(15 * np.pi) / 180, prob=0.1, padding_mode='zeros'),
            monai.transforms.RandAdjustContrast(gamma=(0.5, 1), prob=0.1),
            monai.transforms.ToTensor()
        ])

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((400, 400)),
        ])

        train_ds, valid_ds, test_ds = create_ds(
            args, transform_train, transform)

        train_dataloader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True)
        valid_dataloader = DataLoader(
            valid_ds, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=True)

        loss_fun = get_loss_fun(args)

        optimizer = get_optimizer(model, args)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.learning_rate_max,
            steps_per_epoch=len(train_dataloader), epochs=args.num_epoch)

        train_metrics, valid_metrics = train(model, train_dataloader,
                                             valid_dataloader, loss_fun,
                                             optimizer, scheduler, device, num_epochs=args.num_epoch,
                                             patience=args.pacience, regresion=args.regresion)

        time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        torch.save(
            model, f'./modelos_entrenados/{args.out_name}_{model_depth}_{args.num_epoch}_{time}.pth')

        make_plots(train_metrics, valid_metrics, time)

        plot_predictions(model, train_dataloader, 'Predicciones en Entrenamiento',
                         f'./graficas/pred_train_{time}.png', device)

        plot_predictions(model, valid_dataloader, 'Predicciones en Validacion',
                         f'./graficas/pred_valid_{time}.png', device)

        plot_predictions(model, test_dataloader, 'Predicciones en Test',
                         f'./graficas/pred_test_{time}.png', device)

        print('Finished!!')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Expected config.json file', file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        data = json.load(f)

    args = Args(**data)
    for key, val in vars(args).items():
        print(f'{key}: {val}')
    main(args)
