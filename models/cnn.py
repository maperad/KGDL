import torch
import torch.nn as nn
from torchsummary import summary

from models.base import BaseModel


class CnnBase(BaseModel):

    def __init__(
            self, n_classes: int, input_dim: int, input_channels: int,
            embedding_dim: int, projection_dim: int = None,
            parts: list = ['clf'], dataset: str = None,
            split_type: str = 'one'
    ):
        super().__init__(
            n_classes, embedding_dim, projection_dim, parts
        )

        self.input_dim = input_dim
        self.input_channels = input_channels

        if split_type in ['one', 'phase']:
            self.split_type = split_type
        else:
            raise NotImplementedError(
                f"Split type '{split_type}' not implemented."
            )

        # Select the correct embedding network based on the dataset
        if dataset == 'CWRU':
            self.net_embed = EncCnnCWRU(
                input_channels, input_dim, embedding_dim, split_type
            )
        elif dataset == 'PHM22Rockdrill':
            self.net_embed = EncCnnPHM22Rockdrill(
                embedding_dim, split_type
            )
        elif dataset == 'PHMAP23Spacecraftpropulsion':
            self.net_embed = EncCnnPHMAP23Spacecraftpropulsion(
                input_channels, input_dim, embedding_dim, split_type
            )
        else:
            raise NotImplementedError(
                f'Dataset {dataset} not implemented.'
            )

    def get_summary(self, input):
        device = next(self.parameters()).device
        summary(
            self, (input[0].shape[0], input[0].shape[1]),
            device=device.type
        )

    def _epoch_step_body(
            self, data, device, mode, criterion
    ):
        data = {k: v.to(device) for k, v in data.items()}
        out = self(data['X'], mode)
        if mode == 'clf':
            out_proj, out_clf = None, out
            loss = criterion(out_clf, data['y'])
        elif mode == 'embedding':
            out_proj, out_clf = out, None
            loss = criterion(out_proj, data['y'], data['y_kg_emb'])
        return loss, out_clf, out_proj


class EncCnnCWRU(torch.nn.Module):

    def __init__(self, input_channels, input_dim, embedding_dim, split_type):

        super().__init__()
        if split_type == 'one':
            self.n_parts = 1
            kernel = 12
        elif split_type == 'phase':
            self.n_parts = 2
            kernel = 6

        self.conv = nn.ModuleDict({
            str(k): nn.Sequential(
                nn.Conv1d(input_channels, 64, kernel),
                nn.LeakyReLU(0.5),
                nn.MaxPool1d(kernel_size=kernel, stride=2),
                nn.Dropout1d(p=0.1),
                nn.Conv1d(64, 32, kernel),
                nn.LeakyReLU(0.5),
                nn.MaxPool1d(kernel_size=kernel, stride=2),
                nn.Dropout1d(p=0.1),
                nn.Conv1d(32, 16, kernel),
                nn.LeakyReLU(0.5),
                nn.MaxPool1d(kernel_size=kernel, stride=2),
                nn.Dropout1d(p=0.1),
                nn.Conv1d(16, 8, kernel),
                nn.LeakyReLU(0.5),
                nn.MaxPool1d(kernel_size=kernel, stride=2),
                nn.Dropout1d(p=0.1),
                nn.Flatten()
            )
            for k in range(self.n_parts)
        })

        with torch.no_grad():
            x_in = torch.zeros(1, input_channels, input_dim)
            crop_len = x_in.shape[2] // self.n_parts
            flat_output_dim = self.conv['0'](x_in[:, :, 0:crop_len]).shape[1]
        conv_out_dim = flat_output_dim * self.n_parts

        self.lin_layers = nn.Sequential(
            nn.Linear(conv_out_dim, embedding_dim),
            nn.LeakyReLU(0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        crop_len = x.shape[2] // self.n_parts
        for i in range(self.n_parts):
            start = i * crop_len
            end = (i + 1) * crop_len
            features.append(
                self.conv[str(i)](x[:, :, start:end])
            )
        features = torch.cat(features, dim=1)
        features = self.lin_layers(features)
        return features


class EncCnnPHM22Rockdrill(torch.nn.Module):
    '''
    Implementation based on the following paper:
    @article{kim2023domain,
        title={
            Domain adaptation based fault diagnosis under variable
            operating conditions of a rock drill
        },
        author={
            Kim, Yong Chae and Kim, Taehun and Ko, Jin Uk and Lee, Jinwook
            and Kim, Keon
        },
        journal={International Journal of Prognostics and Health Management},
        volume={14},
        number={2},
        year={2023}
    }
    '''
    def __init__(self, embedding_dim: int, split_type: str):
        super().__init__()
        if split_type == 'one':
            self.n_parts = 1
        elif split_type == 'phase':
            self.n_parts = 3
        self.n_domains = 8

        self.conv = nn.ModuleDict({
            str(k): nn.Sequential(
                nn.Conv1d(
                    3, 16, kernel_size=15, stride=1,
                    padding=7,
                ),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(
                    16, 32, kernel_size=15, stride=1,
                    padding=7,
                ),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(
                    32, 64, kernel_size=7, stride=1, padding=3
                ),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(
                    64, 128, kernel_size=7, stride=1, padding=3,
                ),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(
                    128, 256, kernel_size=3, stride=1, padding=1,
                ),
                nn.Flatten()
            ) for k in range(self.n_parts)
        })

        flat_output_dim = 57600 // self.n_parts  # 19200

        for k in range(self.n_parts):
            self.conv[str(k)].add_module(
                'fc', nn.Sequential(
                    nn.Linear(flat_output_dim, 512),
                    nn.ReLU()
                )
            )

        self.lin_layers = nn.Sequential(
            nn.Linear(self.n_parts*512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        crop_len = x.shape[2] // self.n_parts
        for i in range(self.n_parts):
            start = i * crop_len
            end = (i + 1) * crop_len
            features.append(
                self.conv[str(i)](x[:, :, start:end])
            )
        features = torch.cat(features, dim=1)
        features = self.lin_layers(features)
        return features


class EncCnnPHMAP23Spacecraftpropulsion(EncCnnCWRU):

    def __init__(self, input_channels, input_dim, embedding_dim, split_type):
        super().__init__(input_channels, input_dim, embedding_dim, split_type)
