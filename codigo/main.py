import argparse
from train import train, evaluate_loader
import torch
import torch.nn as nn
import torch.optim as optim
from data_loaders import DataSetMRIs
from torch.utils.data import DataLoader
from datetime import datetime
import torchvision
import torchio as tio
import seaborn as sns
import matplotlib.pyplot as plt
from create_model import ResNet3D_Regresion
from torchsummary import summary
import re
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_train', type=str,
                        help='Folder with the data for training')

    parser.add_argument('--data_valid', type=str,
                        help='Folder with the data for validation')

    parser.add_argument('--data_test', type=str,
                        help='Folder with the data for test')

    parser.add_argument('--mn_model_path', type=str,
                        help='Medical Net model to Use')

    parser.add_argument('--batch', type=int, default=32, help='Batch size')

    parser.add_argument('--num_epochs', type=int,
                        default=25, help='Num Epochs')

    parser.add_argument('--pacience', type=int, default=5,
                        help='Patience of early stopper')

    parser.add_argument('--train', type=bool, default=True,
                        help='If train or evaluate')

    parser.add_argument('--model_path', type=str,
                        help='Path to the model, only used if train is false')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')

    return parser.parse_args()


def make_plots(data_train, data_val, time):
    for item, _ in data_train.items():
        sns.lineplot(data_train[item], label='Train')
        sns.lineplot(data_val[item], label='Valid')

        plt.title(f'Evolución de {item}')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.tight_layout()
        plt.grid(True)
        plt.legend()
        plt.savefig(f'./graficas/{item}_{time}.png', dpi=600)
        plt.close()


def load_pretrained_model(pretrain_path):
    match = re.search(r"resnet_(\d+)", pretrain_path)

    if match:
        model_depth = int(match.group(1))
    else:
        print("Model depth not found, please the format: rester_dethp*",
              file=sys.stderr)

        return None

    model = ResNet3D_Regresion(model_depth)
    checkpoint = torch.load(pretrain_path)
    state_dict = checkpoint['state_dict']
    state_dict = {k.replace('module.', 'model.')
                            : v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    print('-' * 50)
    print('Summary for entrance of size (1, 20, 100, 100)')
    print(summary(model, (1, 20, 100, 100)))
    print('-' * 50)

    return model


def main(args):
    train_folder = args.data_train
    valid_folder = args.data_valid
    test_folder = args.data_test
    batch_size = args.batch
    num_epoch = args.num_epochs
    pacience = args.pacience
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        model = load_pretrained_model(args.mn_model_path)

        model.to(device)

        transform = tio.Compose([
            tio.Resample((1.0, 1.0, 1.0)),
            tio.CropOrPad((25, 350, 350))
        ])

        train_ds = DataSetMRIs(train_folder, transform=transform)
        valid_ds = DataSetMRIs(valid_folder, transform=transform)
        test_ds = DataSetMRIs(test_folder, transform=transform)

        train_dataloader = DataLoader(train_ds, batch_size=batch_size)
        valid_dataloader = DataLoader(valid_ds, batch_size=batch_size)
        test_dataloader = DataLoader(test_ds, batch_size=batch_size)

        loss_fun = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        train_metrics, valid_metrics = train(model, train_dataloader,
                                             valid_dataloader, loss_fun,
                                             optimizer, device, num_epochs=num_epoch,
                                             patience=pacience)

        time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        torch.save(model, f'./modelos_entrenados/model_{time}.pth')

        make_plots(train_metrics, valid_metrics, time)

        print('Finished!!')


if __name__ == '__main__':
    args = parse_args()
    main(args)
