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


def adapt_model(model):
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


def main(args):
    train_folder = args.data_train
    valid_folder = args.data_valid
    test_folder = args.data_test
    batch_size = args.batch
    num_epoch = args.num_epochs
    pacience = args.pacience

    if args.train:
        model = ResNet3D_Regresion(
            layers=[1, 1, 1, 1],
            sample_input_D=64,
            sample_input_H=128,
            sample_input_W=128,
            num_output=1,
            shortcut_type='B'
        )

        transform = torchvision.transforms.Compose([
            tio.Resize((32, 128, 128)),
        ])

        train_ds = DataSetMRIs(train_folder, transform=transform)
        valid_ds = DataSetMRIs(valid_folder, transform=transform)
        test_ds = DataSetMRIs(test_folder, transform=transform)

        train_dataloader = DataLoader(train_ds, batch_size=batch_size)
        valid_dataloader = DataLoader(valid_ds, batch_size=batch_size)
        test_dataloader = DataLoader(test_ds, batch_size=batch_size)

        loss_fun = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
