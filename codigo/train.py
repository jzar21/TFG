import torch
import torchvision
from torch.utils.data import DataLoader
import monai
from data_loaders import DataSetMRIs
import sys
import torchio as tio
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, \
    f1_score, recall_score, precision_score, accuracy_score
from early_stopper import EarlyStopping
import gc


def train_one_epoch(model, dataloader_train,
                    optimizer, loss_function, scheduler, device):
    model.train()

    for i, (im, label, metadata) in enumerate(dataloader_train):
        im, label, metadata = im.to(device), label.to(
            device), metadata.to(device)

        optimizer.zero_grad()
        outputs = model((im, metadata)).view(-1)
        loss = loss_function(outputs, label)
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(
            f'\033[32mBatch[{i}/{len(dataloader_train)}] loss: {loss.item():.4f}\033[0m')


def get_predictions(model, dataloader, device):
    model.eval()
    predicted = []
    reals = []
    for im, label in dataloader:
        im, label = im.to(device), label.to(device)
        with torch.no_grad():
            prediction = model(im).view(-1).detach().cpu().numpy().tolist()
            real = label.cpu().numpy().tolist()

        predicted.extend(prediction)
        reals.extend(real)

    return np.array(predicted), np.array(reals)


def evaluate_loader_regresion(model, dataloader, device):
    metrics = {}
    predicted, reals = get_predictions(model, dataloader, device)
    metrics['MSE'] = mean_squared_error(reals, predicted)
    metrics['MAE'] = mean_absolute_error(reals, predicted)
    metrics['R2'] = r2_score(reals, predicted)

    return metrics


def evaluate_loader_classification(model, dataloader, device):
    metrics = {}
    loss_fn = torch.nn.BCELoss()
    predicted, reals = get_predictions(model, dataloader, device)
    loss_value = loss_fn(torch.tensor(predicted), torch.tensor(reals))

    predicted = (predicted > 0.5).astype(float)
    metrics['Precision'] = precision_score(reals, predicted)
    metrics['Recall'] = recall_score(reals, predicted)
    metrics['Accuracy'] = accuracy_score(reals, predicted)
    metrics['F1'] = f1_score(reals, predicted)
    metrics['BCE'] = loss_value.item()

    return metrics


def evaluate_loader(model, data_loader, device, regresion):
    if regresion:
        return evaluate_loader_regresion(model, data_loader, device)

    return evaluate_loader_classification(model, data_loader, device)


def create_dicts(regresion):
    train_metrics = None
    valid_metrics = None
    if regresion:
        train_metrics = {'MSE': [], 'MAE': [], 'R2': []}
        valid_metrics = {'MSE': [], 'MAE': [], 'R2': []}
    else:
        train_metrics = {'Precision': [],
                         'Recall': [], 'Accuracy': [], 'BCE': [], 'F1': []}
        valid_metrics = {'Precision': [],
                         'Recall': [], 'Accuracy': [], 'BCE': [], 'F1': []}

    return train_metrics, valid_metrics


def get_last_loss(dict, regresion):
    if regresion:
        return dict['MSE'][-1]

    return dict['BCE'][-1]


def train(model, train_loader, valid_loader, loss_function, optimizer, scheduler, device,
          num_epochs=25, patience=5, regresion=True):

    train_metrics, valid_metrics = create_dicts(regresion)
    early_stoper = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        print("-" * 50)
        print(f"Epoch {epoch + 1}/{num_epochs}")

        train_one_epoch(model, train_loader, optimizer,
                        loss_function, scheduler, device)

        train_evaluation = evaluate_loader(
            model, train_loader, device, regresion)
        valid_evaluation = evaluate_loader(
            model, valid_loader, device, regresion)

        for item, values in train_evaluation.items():
            print(f'Train {item}: {values}')
            train_metrics[item].append(values)

        for item, values in valid_evaluation.items():
            print(f'Valid {item}: {values}')
            valid_metrics[item].append(values)

        last_valid_loss = get_last_loss(valid_metrics, regresion)

        if early_stoper(last_valid_loss, model):
            print('Early stopping!!')
            early_stoper.load_best_model(model)
            break

        gc.collect()

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
