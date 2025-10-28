from itertools import product
import torch
from torch.utils.data import DataLoader, Dataset
import logging
from pathlib import Path
from itertools import compress
import copy

from torch_geometric.data import HeteroData, Batch
from torch_geometric.loader import DataLoader as GraphDataLoader

from sklearn.model_selection import train_test_split

from knowledge_graph import KnowledgeGraph


class DataPrep():
    def __init__(self, force_reload: bool = False, seed: int = None):
        self.FORCE_RELOAD = force_reload

        self.X = None
        self.y = None
        self.n_train_classes = None
        self.y_score = None
        self.y_kg_emb = None
        self.y_hierarchy_dict = None
        self.ood_faults = []
        self.seed = seed
        self.setting_name = None

    def get_euclidean_data(self, *args, **kwargs):
        raise NotImplementedError

    def _reorder_labels_ood_faults(self, ood_faults=[]):
        if len(ood_faults) == 0:
            return
        new_target_map = copy.deepcopy(self.y_target_map)
        for c in ood_faults:
            new_target_map.pop(c)
        new_target_map = {
            k: i for i, k in enumerate(new_target_map.keys())
        }
        for c in ood_faults:
            new_target_map[c] = len(new_target_map)
        y_target_map_switched = {
            v: k for k, v in new_target_map.items()
        }
        self.y = torch.tensor([
            new_target_map[y_target_map_switched[y_i.item()]]
            for y_i in self.y
        ])
        new_y = torch.empty_like(self.y)
        for k, v in self.y_target_map.items():
            new_y[self.y == v] = new_target_map[k]
        self.y = new_y
        self.y_target_map = new_target_map

    def save_euclidean_data(self, data_dict: dict, data_path: Path) -> None:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data_dict, data_path)
        logging.info("Save data: %s.", data_path)

    def load_euclidean_data(self, data_path: Path) -> dict:
        logging.info("Load data: %s.", data_path)
        return torch.load(data_path, weights_only=False)

    def transform_to_graph_data(
            self, edge_attr: bool = True, graph_type: str = 'hetero',
            split_type: str = 'phase', bandwidth: float = 2.0
    ) -> None:
        raise NotImplementedError

    def save_graphs(
            self, graph_list: list, graph_path: Path,
            graph_ordering: list = None
    ) -> None:
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_batch = Batch.from_data_list(graph_list)
        logging.info("Saving phase graph data to %s", graph_path)
        torch.save({'gb': graph_batch, 'go': graph_ordering}, graph_path)

    def load_graphs(self, graph_path: Path) -> list:
        logging.info("Loading phase graph data from %s", graph_path)
        graph_data = torch.load(graph_path, weights_only=False)
        graph_list = graph_data['gb'].to_data_list()
        graph_ordering = graph_data['go']
        if graph_ordering is not None:
            # Reorder euclidean data to match new ordering
            euc_data_names = self._get_rel_data_names()
            for data_name in euc_data_names:
                attr = getattr(self, data_name)
                if attr is not None:
                    setattr(self, data_name, attr[graph_ordering])

        # Add correct labels from euclidean data and selected kg
        for i, g in enumerate(graph_list):
            g['y'] = self.y[i].reshape(1)
            try:
                g['y_kg_emb'] = self.y_kg_emb[i].reshape(-1)
            except TypeError:
                g['y_kg_emb'] = None
        return graph_list

    def _split_data_by_phase(self, *args, **kwargs):
        raise NotImplementedError

    def gaussian_kernel_weight_dist(
            self, v1: torch.Tensor, v2: torch.Tensor, beta: float
    ) -> torch.Tensor:
        return torch.exp(
            -torch.norm(v1 - v2, dim=1) ** 2 / (2 * beta ** 2)
        )

    def _create_phase_graph(
            self,
            X: dict,
            y: torch.Tensor,
            edge_attr: bool = True,
            bandwidth: float = 2.0,
            graph_type: str = 'hetero',
            additional_info: dict = {}
            ) -> list:
        graph_list = []
        sensors = list(X.keys())
        phases = list(X[sensors[0]].keys())

        # Create a graph for each sample
        for i, y_i in enumerate(y):
            tmp_graph = HeteroData()

            # Add nodes
            for p in phases:
                tmp_graph[p].x = torch.stack([
                    X[s][p][i]
                    for s in sensors
                ], axis=1).T.type(torch.float32)

            # Add edges
            # p1 <-> p1, p2 <-> p2, p3 <-> p3 ... for each sensor
            for p in phases:
                # if len(sensors) > 1:
                s_idx = list(range(len(sensors)))
                s_edges = list(product(s_idx, s_idx))
                if len(sensors) > 1:
                    # Delete self loops
                    # Keep self loops when only 1 sensor,
                    # else first phase has no input connection.
                    # In the Gnn this corresponds to a linear transformation
                    # to the required dimension.
                    s_edges = [e for e in s_edges if e[0] != e[1]]
                tmp_graph[p, f"e_{p}", p].edge_index = \
                    torch.tensor([
                        s_edges
                        # [1, 2, 2, 0, 0, 1],
                        # [0, 0, 1, 1, 2, 2]
                    ], dtype=torch.long).squeeze(dim=0).T
                if edge_attr:
                    p_node = tmp_graph[p].x
                    p_edge_index = tmp_graph[p, f"e_{p}", p].edge_index
                    tmp_graph[p, f"e_{p}", p].edge_attr = \
                        self.gaussian_kernel_weight_dist(
                            p_node[p_edge_index[0]], p_node[p_edge_index[1]],
                            bandwidth
                        ).type(torch.float32).reshape(-1, 1)

            # p1 -> p2 -> p3 ... for each sensor
            for p_i, p in enumerate(phases[:-1]):
                p_next = phases[p_i+1]
                p_edges = list(range(len(sensors)))
                tmp_graph[p, f"e_{p}_{p_next}", p_next].edge_index = \
                    torch.tensor([
                        p_edges, p_edges
                        # [0, 1, 2],
                        # [0, 1, 2]
                    ], dtype=torch.long)
                if edge_attr & (graph_type == 'homo'):
                    tmp_graph[p, f"e_{p}_{p_next}", p_next].edge_attr = \
                        torch.tensor(
                            [1 for _ in p_edges], dtype=torch.float32
                            # [1, 1, 1], dtype=torch.float32
                        ).reshape(-1, 1)

            # Add graph label
            tmp_graph['y'] = y_i.long()
            tmp_graph['n_phases'] = len(phases)

            # Add additional information to graph
            for k, v in additional_info.items():
                try:
                    tmp_graph[k] = v[i].long()
                except TypeError:
                    next

            if graph_type == 'homo':
                # Transform heterogenous graph to homogeneous
                tmp_graph = tmp_graph.to_homogeneous()

            # Add to graph list
            graph_list.append(tmp_graph)

        return graph_list

    def _get_rel_data_names(self) -> list[str]:
        raise NotImplementedError

    def _check_data(self) -> None:
        self.data_names = self._get_rel_data_names()
        for data_name in self.data_names:
            # Check all data attributes are filled, else create empty tensors
            if getattr(self, data_name) is None:
                if data_name.startswith('y'):
                    setattr(self, data_name, torch.empty_like(self.y))
                else:
                    setattr(self, data_name, torch.empty_like(self.X))
            # Convert to long if data is a label and float if data is a feature
            if data_name in ['y', 'y_score', 'rpm', 'setting']:
                setattr(self, data_name, getattr(self, data_name).long())
            else:
                setattr(self, data_name, getattr(self, data_name).float())

    def _train_test_split_size(
            self, data_dict: dict, test_size: float
    ) -> tuple[dict, dict]:
        split_data = train_test_split(
            *[v for v in data_dict.values()], test_size=test_size,
            random_state=self.seed
        )
        train_data = {
            k: split_data[i*2] for i, k in enumerate(data_dict.keys())
        }
        test_data = {
            k: split_data[i*2+1] for i, k in enumerate(data_dict.keys())
        }
        return train_data, test_data

    def _train_test_train_mask(
            self, data_dict: dict, train_mask: torch.tensor
    ) -> tuple[dict, dict]:
        train_data = {
            k: v[train_mask] for k, v in data_dict.items()
            if isinstance(v, torch.Tensor)
        }
        train_data |= {
            k: list(compress(v, train_mask)) for k, v in data_dict.items()
            if isinstance(v, list)
        }
        shuffle_test_idx = torch.randperm(sum(~train_mask))
        # shuffle test data, else SupConLoss is always 0 because all samples
        # in one batch have the same label
        # -> no negative samples for contrastive loss
        test_data = {
            k: v[~train_mask][shuffle_test_idx] for k, v in data_dict.items()
            if isinstance(v, torch.Tensor)
        }
        test_data |= {
            k: list(compress(v, ~train_mask))
            for k, v in data_dict.items() if isinstance(v, list)
        }
        test_data |= {
            k: [v[i] for i in shuffle_test_idx]
            for k, v in test_data.items() if isinstance(v, list)
        }
        return train_data, test_data

    def _train_test_split_ood(
        self, data_dict: dict, ood_fault_list: list
    ) -> tuple[dict, dict]:
        ood_fault_list_int = [
            self.y_target_map[c] for c in ood_fault_list
        ]
        if 'graph_list' in data_dict:
            y_data = torch.tensor(
                [g['y'] for g in data_dict['graph_list']]
            )
        else:
            y_data = data_dict['y']
        ood_idx = torch.isin(y_data, torch.tensor(ood_fault_list_int))
        return self._train_test_train_mask(data_dict, ~ood_idx)

    def _train_test_split_setting(
            self, data_dict: dict, setting: str, train_settings: list
    ) -> tuple[dict, dict]:
        if 'graph_list' in data_dict:
            setting_data = torch.tensor(
                [g[setting] for g in data_dict['graph_list']]
            )
        else:
            setting_data = data_dict[setting]
        train_idx = torch.isin(
            setting_data, torch.tensor(train_settings)
        )
        return self._train_test_train_mask(data_dict, train_idx)

    def get_kg_emb_target(self, kg_path: Path,) -> None:
        kg = KnowledgeGraph(kg_path, self.ood_faults)
        self.y_kg_emb = kg.map_embedding(self.y, self.y_target_map)

    def _worker_init_fn(self, worker_id: int) -> None:
        torch.manual_seed(self.seed + worker_id)

    def create_euclidean_dataloaders(
            self, test_size: float, batch_size: int,
            train_settings: list = None, ood_fault_list: list = None
    ) -> tuple[DataLoader, dict]:
        test_data = {}
        self._check_data()  # get data names and check their existence
        data_dict = {name: getattr(self, name) for name in self.data_names}
        if ood_fault_list is not None:
            train_data, test_data['ood'] = self._train_test_split_ood(
                data_dict, ood_fault_list
            )
            data_dict = train_data

        # Split data by setting
        train_data, test_data['test_setting'] = \
            self._train_test_split_setting(
                data_dict, self.setting_name, train_settings
            )
        data_dict = train_data
        if ood_fault_list is not None:
            test_data['train_setting_ood'], \
                test_data['test_setting_ood'] = \
                self._train_test_split_setting(
                    test_data['ood'], self.setting_name, train_settings
                )
            test_data.pop('ood', None)
        train_data, test_data['train_setting'] = self._train_test_split_size(
            data_dict, test_size
        )
        self.domain_label_map = {
            str(c): i for i, c in enumerate(
                set(train_data[self.setting_name].tolist())
            )
        }
        train_data['y_domain'] = torch.tensor(
            [self.domain_label_map[str(s.item())]
             for s in train_data[self.setting_name]], dtype=torch.long
        )
        dataloader_train = DataLoader(
            DictDataset(train_data), batch_size=batch_size, shuffle=True,
            num_workers=0, worker_init_fn=self._worker_init_fn
        )
        dataloader_test = {}
        for k, v in test_data.items():
            dataloader_test[k] = DataLoader(
                DictDataset(v), batch_size=batch_size, shuffle=False,
                num_workers=0, worker_init_fn=self._worker_init_fn
            )
        return (dataloader_train, dataloader_test)

    def create_graph_dataloaders(
        self, test_size: float, batch_size: int,
        train_settings: list = None, ood_fault_list: list = None
    ) -> tuple[GraphDataLoader, dict]:
        test_graphs = {}
        data_dict = {'graph_list': self.graph_list}
        if (ood_fault_list is not None) and (ood_fault_list != []):
            train_graphs, test_graphs['ood'] = \
                self._train_test_split_ood(data_dict, ood_fault_list)
            data_dict = train_graphs

        train_graphs, test_graphs['test_setting'] = \
            self._train_test_split_setting(
                data_dict, self.setting_name, train_settings
            )
        data_dict = train_graphs
        if (ood_fault_list is not None) and (ood_fault_list != []):
            test_graphs['train_setting_ood'], \
                test_graphs['test_setting_ood'] = \
                self._train_test_split_setting(
                    test_graphs['ood'], self.setting_name, train_settings
                )
            test_graphs.pop('ood', None)
        train_graphs, test_graphs['train_setting'] = \
            self._train_test_split_size(data_dict, test_size)
        dataloader_train = GraphDataLoader(
            train_graphs['graph_list'], batch_size=batch_size, shuffle=True,
            num_workers=0, worker_init_fn=self._worker_init_fn
        )
        dataloader_test = {}
        for k, v in test_graphs.items():
            dataloader_test[k] = GraphDataLoader(
                v['graph_list'], batch_size=batch_size, shuffle=False,
                num_workers=0, worker_init_fn=self._worker_init_fn
            )
        return dataloader_train, dataloader_test

    def _y_str_to_int_label_mapping(self, y: torch.Tensor) -> dict:
        return {str(c): i for i, c in enumerate(set(y))}

    def _y_str_to_int(self, y: torch.Tensor, y_map: dict) -> torch.Tensor:
        return torch.tensor([y_map[c] for c in y], dtype=torch.long)


class DictDataset(Dataset):
    '''
    Dictionary dataset based on
    https://discuss.pytorch.org/t/dictionary-in-dataloader/40448/2
    '''
    def __init__(self, data_dict):
        if not all(
            len(v) == len(list(data_dict.values())[0])
            for v in data_dict.values()
        ):
            raise ValueError(
                "All values in the dictionary must have the same length."
            )
        for k, v in data_dict.items():
            setattr(self, k, v)

    def __getitem__(self, index):
        return_dict = {
            k: v[index]
            for k, v in self.__dict__.items()
        }
        return return_dict

    def __len__(self):
        return len(list(self.__dict__.values())[0])


def create_emb_dataloader(
        emb, y, batch_size: int, shuffle: bool = True
) -> DataLoader:
    return DataLoader(
        DictDataset({'X': emb, 'y': y}),
        batch_size=batch_size, shuffle=shuffle
    )
