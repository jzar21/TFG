import torch
import torchvision
from torch.utils.data import DataLoader
import monai
from data_loaders import DataSetMRIs
import sys
import torchio as tio
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from early_stopper import EarlyStopping


def train_one_epoch(model, dataloader_train,
                    optimizer, loss_function, device, verbose_percentaje=0.3):
    model.train()
    percentage_info = int(verbose_percentaje * len(dataloader_train))
    avg_batch_loses = []

    for i, (im, label) in enumerate(dataloader_train):
        im, label = im.to(device), label.to(device)

        optimizer.zero_grad()
        outputs = model(im).view(-1)  # (batch_size, 1) a (batch_size)
        loss = loss_function(outputs, label)
        loss.backward()
        optimizer.step()

        avg_batch_loses.append(loss.item() / im.shape[0])

        # if i % percentage_info == 0:
        print(
            f'\033[32mBatch[{i}/{len(dataloader_train)}] loss: {loss.item():.4f}\033[0m')

    return np.array(avg_batch_loses)


def evaluate_loader(model, dataloader, device):
    model.eval()
    metrics = {}
    predicted = []
    reals = []
    for im, label in dataloader:
        im, label = im.to(device), label.to(device)
        predicted.extend(model(im).view(-1).detach().cpu().numpy().tolist())
        reals.extend(label.cpu().numpy().tolist())

    metrics['MSE'] = mean_squared_error(reals, predicted)
    metrics['MAE'] = mean_absolute_error(reals, predicted)
    metrics['R2'] = r2_score(reals, predicted)

    return metrics


def train(model, train_loader, valid_loader, loss_function, optimizer, device,
          num_epochs=25, patience=5, verbose_percent=0.3):

    train_metrics = {'MSE': [], 'MAE': [], 'R2': []}
    valid_metrics = {'MSE': [], 'MAE': [], 'R2': []}
    early_stoper = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        print("-" * 50)
        print(f"Epoch {epoch + 1}/{num_epochs}")

        avg_batch_loses = train_one_epoch(model, train_loader, optimizer,
                                          loss_function, device, verbose_percentaje=verbose_percent)

        train_evaluation = evaluate_loader(model, train_loader, device)
        valid_evaluation = evaluate_loader(model, valid_loader, device)

        for item, values in train_evaluation.items():
            print(f'Train {item}: {values}')
            train_metrics[item].append(values)

        for item, values in valid_evaluation.items():
            print(f'Valid {item}: {values}')
            valid_metrics[item].append(values)

        if early_stoper(valid_metrics['MSE'][-1], model):
            print('Early stopping!!')
            early_stoper.load_best_model(model)
            break

    print(f"Train completed")

    return train_metrics, valid_metrics


def main(data_folder: str):
    transform = torchvision.transforms.Compose([
        tio.Resize((32, 128, 128)),
    ])

    data_set = DataSetMRIs(data_folder, transform=transform)
    data_loader = DataLoader(data_set, batch_size=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.DenseNet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
    ).to(device)

    train(model, data_loader, data_set, device)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Error:\nExpexted: {sys.argv[0]} <data_dir>", file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1])
