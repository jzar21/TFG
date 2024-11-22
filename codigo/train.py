import torch
import torchvision
from torch.utils.data import DataLoader
import monai
from data_loaders import DataSetMRIs
import sys
import torchio as tio


def train(model, train_loader, train_ds, device):
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    epoch_loss_values = list()

    for epoch in range(5):
        print("-" * 50)
        print(f"epoch {epoch + 1}/{5}")
        model.train()
        epoch_loss = 0
        step = 0
        for im, label in train_loader:
            step += 1
            im, label = im.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(im)
            loss = loss_function(outputs, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    print(f"train completed")


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
