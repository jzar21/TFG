from dataclasses import dataclass, field


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
    fc_layers_arch: list = field(default_factory=lambda: [1024, 512, 256, 1])
    flip_prob: float = 0.2
    perspective_prob: float = 0.2
    rot_degree: float = 10.0
    rot_prob: float = 0.2
    contrast_gamma: tuple = (0.5, 1)
    contrast_prob: float = 0.2
