from dataclasses import dataclass


@dataclass
class Args:
    train_folder: str = './data'
    valid_folder: str = './data'
    test_folder: str = './data'
    batch_size: int = 16
    num_epoch: int = 50
    pacience: int = 15
    model_path: str = './model/resnet_101.pth'
    learning_rate: float = 0.001
    learning_rate_max: float = 0.01
    from_scratch: bool = False
    num_slices: int = 17
    train: bool = True
    regresion: bool = True
    classification: bool = not regresion
    loss: str = 'MSE'
    optimizer: str = 'adamw'
    out_name: str = 'resnet'
    img_size: tuple = (400, 400)
    use_data_aug: bool = False
    pretrain_med_net: bool = True
