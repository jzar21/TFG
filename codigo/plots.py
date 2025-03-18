import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import torch

plt.style.use(['science', 'ieee', 'grid', 'no-latex'])


def make_plots(data_train, data_val, time):
    for item, _ in data_train.items():
        plt.plot(data_train[item], label='Train')
        plt.plot(data_val[item], label='Valid')

        if item == 'R2':
            plt.title('Evolucion de $R^2$')
        else:
            plt.title(f'Evolucion de {item}')

        plt.xlabel('Epocas')
        # plt.ylabel('Pérdida')
        plt.tight_layout()
        # plt.grid(True)
        plt.legend(loc='best')
        plt.savefig(f'./graficas/{item}_{time}.png', dpi=600)
        plt.close()


def plot_predictions(model, dataloader, title, save_path, device):
    model.eval()
    predicted = []
    reals = []
    x = np.arange(7, 27)
    for im, label in dataloader:
        im, label = im.to(device), label.to(device)

        with torch.no_grad():
            predicted.extend(
                model(im).view(-1).detach().cpu().numpy().tolist())
            reals.extend(label.cpu().numpy().tolist())

    plt.scatter(reals, predicted, s=8)
    plt.plot(x, x, color='red', label='Prediccion perfecta', ls='--')
    plt.ylabel('Prediciones')
    plt.xlabel('Reales')
    plt.title(title)
    plt.tight_layout()
    plt.legend(loc='best')
    plt.savefig(save_path, dpi=600)
    plt.close()
