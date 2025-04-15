import torch
import torch.nn as nn
import torch_geometric.nn as geo_nn

from models.base import BaseModel


class GnnBase(BaseModel):

    def __init__(
            self, n_classes: int,
            embedding_dim: int = 64, projection_dim: int = None,
            n_sensors: int = 3, n_phases: int = 3,
            parts: list = ['clf'], dataset: str = None,
            cnn_pre: bool = True, graph_type: str = 'hetero'
    ):
        super().__init__(
            n_classes, embedding_dim, projection_dim, parts
        )

        # Select the correct embedding network based on the dataset
        if dataset == 'CWRU':
            self.net_embed = EncGnnCWRU(
                n_sensors, n_phases, embedding_dim, cnn_pre,
                graph_type
            )
        elif dataset == 'PHM22Rockdrill':
            self.net_embed = EncGnnPHM22Rockdrill(
                n_sensors, n_phases, embedding_dim, cnn_pre,
                graph_type
            )
        elif dataset == 'PHMAP23Spacecraftpropulsion':
            self.net_embed = EncGnnPHMAP23Spacecraftpropulsion(
                n_sensors, n_phases, embedding_dim, cnn_pre,
                graph_type
            )
        else:
            raise NotImplementedError(
                f'Dataset {dataset} not implemented.'
            )

    def get_summary(self, input):
        input.to(next(self.parameters()).device)
        geo_nn.summary(self, input, max_depth=5)
        # Needs to be done twice, else it doesn't print the complete summary.
        print(geo_nn.summary(self, input, max_depth=5))

    def _epoch_step_body(
            self, data, device, mode, criterion
    ):
        data.to(device)
        out = self(data, mode)

        if isinstance(data['y_kg_emb'], torch.Tensor):
            # Correct kg embedding shape
            batch_size = len(data.y)
            data['y_kg_emb'] = data['y_kg_emb'].reshape(batch_size, -1).float()

        if mode == 'clf':
            out_proj, out_clf = None, out
            loss = criterion(out_clf, data['y'])
        elif mode == 'embedding':
            out_proj, out_clf = out, None
            loss = criterion(out_proj, data['y'], data['y_kg_emb'])

        return loss, out_clf, out_proj


class EncGnnBase(torch.nn.Module):

    def __init__(
            self, n_sensors: int, n_phases: int, embedding_dim: int,
            cnn_pre: bool = True, graph_type: str = 'hetero'
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.n_phases = n_phases
        self.embedding_dim = embedding_dim
        self.cnn_pre = cnn_pre
        self.graph_type = graph_type

        # Parts of the GNN, which are initialized in the subclasses
        self.conv_pre_mod = None
        self.gnn_1 = None
        self.gnn_1_act = None
        self.gnn_2 = None
        self.gnn_2_act = None
        self.lin_embed = None

    def init_hetero_phase_layer(
            self, layer_in: int = -1, layer_out: int = 128
    ):
        gnn_sensor_con = {
            (f'p{p}', f'e_p{p}', f'p{p}'): geo_nn.conv.GATv2Conv(
                layer_in, layer_out, edge_dim=1,
                add_self_loops=False, share_weights=False
            ) for p in range(1, self.n_phases+1)
        }
        gnn_phase_con = {
            (f'p{p}', f'e_p{p}_p{p+1}', f'p{p+1}'): geo_nn.conv.SAGEConv(
                layer_in, layer_out
            ) for p in range(1, self.n_phases)
        }
        gnn_layer = geo_nn.HeteroConv(
            {**gnn_sensor_con, **gnn_phase_con},
            aggr='sum'
        )
        gnn_layer_act = nn.ModuleDict({
            k: nn.PReLU() for k in [f'p{p}' for p in range(1, self.n_phases+1)]
        })
        return gnn_layer, gnn_layer_act

    def forward(self, graph_batch):
        # Heterogeneous graph input
        if self.cnn_pre:
            # Cnn preprocessing module for each phase and sensor
            x_phase_sensor = {
                k: {
                    str(s): self.conv_pre_mod[k][str(s)](
                        graph_batch.x_dict[k][
                            s::self.n_sensors, None, :
                        ]
                    )
                    for s in range(self.n_sensors)
                } for k in [f'p{p}' for p in range(1, self.n_phases+1)]
            }
            # Combine separate forward passes to recreate original
            # graph structure
            x_dict_pre = {}
            for p, v_p in x_phase_sensor.items():
                tmp = torch.empty((
                    v_p['0'].shape[0]*self.n_sensors,
                    v_p['0'].shape[1]
                )).to((next(self.parameters()).device))
                for s, v_s in v_p.items():
                    tmp[int(s)::self.n_sensors, :] = v_s
                x_dict_pre[p] = tmp
        else:
            x_dict_pre = graph_batch.x_dict

        # GNN module
        graph_in = [
            graph_batch.edge_index_dict, graph_batch.edge_attr_dict
        ]
        x_dict_1 = self.gnn_1(x_dict_pre, *graph_in)
        x_dict_1 = {
            k: self.gnn_1_act[k](v)
            for k, v in x_dict_1.items()
        }
        x_dict_2 = self.gnn_2(x_dict_1, *graph_in)
        x_dict_2 = {
            k: self.gnn_2_act[k](v)
            for k, v in x_dict_2.items()
        }
        gnn_out = []
        for p in x_dict_2.keys():
            gnn_out.append(geo_nn.global_add_pool(
                x_dict_2[p],
                batch=graph_batch[p].batch
            ))

        # Concatenate output of GNN
        graph_emb = torch.cat(gnn_out, dim=1)

        return self.lin_embed(graph_emb)


class EncGnnCWRU(EncGnnBase):

    def __init__(
            self, n_sensors: int, n_phases: int, embedding_dim: int,
            cnn_pre: bool = True, graph_type: str = 'hetero'
    ):
        super().__init__(
            n_sensors, n_phases, embedding_dim, cnn_pre, graph_type
        )

        if self.cnn_pre:
            self.conv_pre_mod = nn.ModuleDict({
                f'p{p}': nn.ModuleDict({
                    str(s): self.init_preprocessing_cnn(
                        in_channels=1
                    )
                    for s in range(self.n_sensors)
                }) for p in range(1, self.n_phases+1)
            })

        self.gnn_1, self.gnn_1_act = self.init_hetero_phase_layer(
            layer_in=-1, layer_out=128
        )

        self.gnn_2, self.gnn_2_act = self.init_hetero_phase_layer(
            layer_in=128, layer_out=64
        )

        self.lin_embed = nn.Sequential(
            nn.Linear(self.n_phases*64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, self.embedding_dim)
        )

    def init_preprocessing_cnn(
            self, in_channels: int = 1
            ):
        conv_layers = [
            nn.Conv1d(in_channels, 64, 12),
            nn.Conv1d(64, 32, 12),
            nn.Conv1d(32, 16, 12),
            nn.Conv1d(16, 8, 12)
        ]
        layers = []
        for i, conv in enumerate(conv_layers):
            layers.append(conv)
            layers.append(nn.LeakyReLU(0.5))
            layers.append(nn.MaxPool1d(kernel_size=12, stride=2))
            layers.append(nn.Dropout1d(p=0.1))
        layers.append(nn.Flatten())
        return nn.Sequential(*layers)


class EncGnnPHM22Rockdrill(EncGnnBase):

    def __init__(
            self, n_sensors: int, n_phases: int, embedding_dim: int,
            cnn_pre: bool = True, graph_type: str = 'hetero'
    ):
        super().__init__(
            n_sensors, n_phases, embedding_dim, cnn_pre, graph_type)

        if self.cnn_pre:
            n_pooling_layers = [3, 1, 3]
            self.conv_pre_mod = nn.ModuleDict({
                f"p{p}": nn.ModuleDict({
                    str(s): self.init_preprocessing_cnn(
                        in_channels=1, n_pooling_layers=n_pooling_layers[p-1]
                    )
                    for s in range(self.n_sensors)
                }) for p in range(1, self.n_phases+1)
            })

        self.gnn_1, self.gnn_1_act = self.init_hetero_phase_layer(
            layer_in=-1, layer_out=512
        )

        self.gnn_2, self.gnn_2_act = self.init_hetero_phase_layer(
            layer_in=512, layer_out=256
        )

        self.lin_embed = nn.Sequential(
            nn.Linear(self.n_phases*256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU()
        )

    def init_preprocessing_cnn(
            self, in_channels, n_pooling_layers=3
            ):
        # Check if n_pooling_layers is valid
        try:
            assert n_pooling_layers in [0, 1, 2, 3]
        except AssertionError:
            raise ValueError(
                "n_pooling_layers must be 0, 1, 2, or 3"
            )
        layers = [
            nn.Conv1d(in_channels, 32, 5),
            nn.ReLU()
        ]
        if n_pooling_layers == 3:
            layers.append(nn.MaxPool1d(2))
        layers.extend([
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 16, 5),
            nn.ReLU()
        ])
        if n_pooling_layers >= 2:
            layers.append(nn.MaxPool1d(2))
        layers.extend([
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 8, 5),
            nn.ReLU()
        ])
        if n_pooling_layers >= 1:
            layers.append(nn.MaxPool1d(2))
        layers.extend([
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 1, 5),
            nn.ReLU(),
            nn.BatchNorm1d(1)
        ])
        layers.append(nn.Flatten())
        return nn.Sequential(*layers)


class EncGnnPHMAP23Spacecraftpropulsion(EncGnnBase):

    def __init__(
            self, n_sensors: int, n_phases: int, embedding_dim: int,
            cnn_pre: bool = True, graph_type: str = 'hetero'
    ):
        super().__init__(
            n_sensors, n_phases, embedding_dim, cnn_pre, graph_type
        )

        if self.cnn_pre:
            self.conv_pre_mod = nn.ModuleDict({
                f"p{p}": nn.ModuleDict({
                    str(s): self.init_preprocessing_cnn_open(in_channels=1)
                    for s in range(self.n_sensors)
                }) for p in range(1, self.n_phases+1, 2)
            })
            self.conv_pre_mod.update({
                f"p{p}": nn.ModuleDict({
                    str(s): self.init_preprocessing_cnn_close(
                        in_channels=1
                    )
                    for s in range(self.n_sensors)
                }) for p in range(2, self.n_phases+1, 2)
            })

        if self.graph_type == 'hetero':
            self.gnn_1, self.gnn_1_act = self.init_hetero_phase_layer(
                layer_in=-1, layer_out=128
            )
            self.gnn_2, self.gnn_2_act = self.init_hetero_phase_layer(
                layer_in=128, layer_out=64
            )

        elif self.graph_type == 'homo':
            pass

        self.lin_embed = nn.Sequential(
            nn.Linear(self.n_phases*64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, self.embedding_dim)
        )

    def init_preprocessing_cnn_open(self, in_channels=1):

        layers = [
            nn.Conv1d(in_channels, 32, 5),
            nn.ReLU(),
            # nn.MaxPool1d(2),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 16, 5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 8, 5),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Flatten()
        ]

        return nn.Sequential(*layers)

    def init_preprocessing_cnn_close(self, in_channels=1):

        layers = [
            nn.Conv1d(in_channels, 32, 5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 16, 5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 8, 5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(8),
            nn.Flatten()
        ]

        return nn.Sequential(*layers)
