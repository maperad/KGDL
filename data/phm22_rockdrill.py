import torch
from pathlib import Path
import re

import numpy as np
import pandas as pd

from data.base import DataPrep


class PHM22Rockdrill(DataPrep):
    def __init__(
            self, path_to_data: str, force_reload: bool = False,
            kg_name: str = None, seed: int = None
    ):
        super().__init__(force_reload, seed)

        self.kg_name = kg_name.split('/')[-1]
        self.setting_name = 'setting'
        self.path_to_data = Path(path_to_data)
        self.setting_map = {
            1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7
        }

        self.FAULT_MAPPING = {
            0: ['Normal', 'No-fault'],
            1: ['T', 'Thicker drill steel.'],
            2: ['A',
                'A-seal missing. Leakage from high pressure channel to control'
                + ' channel.'],
            3: ['B',
                'B-seal missing. Leakage from control channel to return'
                + ' channel.'],
            4: ['R', 'Return accumulator, damaged.'],
            5: ['S', 'Longer drill steel.'],
            6: ['D', 'Damper orifice is larger than usual.'],
            7: ['Q', 'Low flow to the damper circuit.'],
            8: ['V', 'Valve damage. A small wear-flat on one of the valve'
                + ' lands.'],
            9: ['O', 'Orifice on control line outlet larger than usual.'],
            10: ['C', 'Charge level in high pressure accumulator is low.']
        }

    def _load_raw_data(self, path, data_type='training'):

        data_dir = path / data_type

        files = sorted(list(data_dir.glob('*.csv')))
        data = {}

        for file in files:
            data[file.stem] = {}
            # Needed because all rows have different length
            df_tmp = pd.read_csv(
                file, header=None, sep='/n', engine='python'
            )[0].str.split(',', expand=True)
            pattern_match = re.search(r'_([a-zA-Z]+)(\d+)$', file.stem)
            data[file.stem]['X'] = df_tmp.iloc[:, 1:].values.astype(float)
            data[file.stem]['Sensor'] = pattern_match.group(1)
            data[file.stem]['Setting'] = int(pattern_match.group(2))

            if 'data' in file.stem:
                file_ans = file.parent.parent / 'answers' / \
                    f"ans{data[file.stem]['Setting']}.csv"
                data[file.stem]['y'] = pd.read_csv(
                    file_ans, header=None
                ).values.squeeze() - 1
            elif 'ref' in file.stem:
                data[file.stem]['y'] = df_tmp.iloc[:, 0].values.astype(int) - 1
            else:
                raise ValueError(f'Unknown file: {file}')

        return data

    def _stack_signals(
            self, data_dict, sensors=['pdmp', 'pin', 'po'], max_cols=None
    ):
        data_sensor_ind = [
            k for k in data_dict.keys()
            if any(sensor in k for sensor in sensors)
        ]
        X = {k: [] for k in sensors}
        y = []
        settings = []
        for dsi in data_sensor_ind:
            y.append(data_dict[dsi]['y'])
            settings.extend(
                [data_dict[dsi]['Setting']] * len(data_dict[dsi]['y'])
            )
            for s in sensors:
                tmp_key = f"{dsi.split('_')[0]}_{s}{settings[-1]}"
                X[s].append(data_dict[tmp_key]['X'])
        y = np.concatenate(y)
        settings = np.array(settings)

        # Pad or truncate signals to same length and stack to one array
        # per sensor
        for k, v in X.items():
            if max_cols is None:
                max_cols = max([sig.shape[1] for sig in v])
            res = []
            for sig in v:
                if max_cols > sig.shape[1]:
                    # Pad signals
                    padded_sig = np.pad(
                        sig,
                        ((0, 0), (0, max_cols - sig.shape[1])),
                        mode='constant',
                        constant_values=np.nan
                    )
                else:
                    # Truncate signals
                    padded_sig = sig[:, :max_cols]
                res.append(padded_sig)
            X[k] = np.vstack(res)

        return X, y, settings

    def load(
            self, path, data_type='training', max_cols=None, force_reload=False
    ):
        saved_dir = path / data_type / f'saved_{max_cols}'
        if saved_dir.exists() and not force_reload:
            X = {
                s: np.load(saved_dir / f'X_{s}.npy')
                for s in ['pdmp', 'pin', 'po']
            }
            y = np.load(saved_dir / 'y.npy')
            settings = np.load(saved_dir / 'settings.npy')
            return X, y, settings

        data = self._load_raw_data(path, data_type)
        X, y, settings = self._stack_signals(data, max_cols=max_cols)

        saved_dir.mkdir(parents=True, exist_ok=True)
        for s in X.keys():
            np.save(saved_dir / f'X_{s}.npy', X[s])
        np.save(saved_dir / 'y.npy', y)
        np.save(saved_dir / 'settings.npy', settings)

        return X, y, settings

    def _find_acceleration_phases(
            self, X_po: torch.Tensor, setting_ref: torch.Tensor, inds: list
    ) -> tuple[dict, dict]:
        accel_rear_end = {}
        accel_for_start = {}
        for i in inds:
            accel_rear_end[i] = torch.argmax(
                X_po[setting_ref == i, :].mean(axis=0)
            ).item()
            accel_for_start[i] = accel_rear_end[i] + 100
        return accel_rear_end, accel_for_start

    def _split_data_by_phase(
            self, X: torch.Tensor, y: torch.Tensor, setting: torch.Tensor,
            individuals: list,
            accel_rear_end: dict = None, accel_for_start: dict = None
    ) -> tuple[dict, torch.Tensor, torch.Tensor]:
        if (accel_rear_end is None) and (accel_for_start is None):
            # No phase splitting -> simple sensor graph
            X_phases_ref = {
                s: {
                    'p1': {
                        i: X[setting == i, j, :]
                        for i in individuals
                    }
                } for j, s in enumerate(self.sensors)
            }
            y_phases_ref = torch.concatenate([
                y[setting == i] for i in individuals
            ])
            new_ordering_list = [
                (setting == i).nonzero(as_tuple=True)[0].tolist()
                for i in individuals
            ]
            new_ordering_idx = [
                item
                for sublist in new_ordering_list
                for item in sublist
            ]
            return X_phases_ref, y_phases_ref, new_ordering_idx

        # Split data into phases according to ref data
        phases = ['p1', 'p2', 'p3']
        X_phases_ref = {
            s: {
                p: {
                    i: [] for i in individuals
                } for p in phases
            } for s in self.sensors
        }
        y_phases_ref = []
        new_ordering_idx = []  # for reordering original data

        for i in individuals:
            phase1 = accel_rear_end[i]
            phase2 = accel_for_start[i]
            i_setting = (setting == i)
            new_ordering_idx.extend(
                i_setting.nonzero(as_tuple=True)[0].tolist()
            )
            for j, s in enumerate(self.sensors):
                X_phases_ref[s]['p1'][i] = X[i_setting, j, :phase1]
                X_phases_ref[s]['p2'][i] = X[i_setting, j, phase1:phase2]
                X_phases_ref[s]['p3'][i] = X[i_setting, j, phase2:]
            y_phases_ref.append(y[i_setting])
        y_phases_ref = torch.concatenate(y_phases_ref)

        return X_phases_ref, y_phases_ref, new_ordering_idx

    def _align_arrays(
            self, arrays: list, stack: bool = True,
            fill_value: float = 0, max_cols: int = None
    ) -> torch.Tensor:
        if max_cols is None:
            # Find the maximum number of columns among all arrays
            max_cols = max(array.shape[-1] for array in arrays)

        # Pad each array to have the same number of columns
        out_array = []
        for array in arrays:
            if array.shape[-1] < max_cols:
                padded_array = torch.full(
                    (*array.shape[:-1], max_cols), fill_value,
                    dtype=array.dtype, device=array.device
                )
                padded_array[..., :array.shape[-1]] = array
                out_array.append(padded_array)
            else:
                out_array.append(array[..., :max_cols])

        # Stack the padded arrays along the first dimension
        if stack:
            return torch.concatenate(out_array, axis=0)
        else:
            return out_array

    def _align_phase_data(
            self, X_phases_ref: dict, max_cols=None
            ):
        sensors = list(X_phases_ref.keys())
        phases = list(X_phases_ref[sensors[0]].keys())
        X_phases_ref_padded = {
            s: {
                p: {} for p in phases
            } for s in sensors
        }
        for phase in phases:
            for sensor in sensors:
                for ind in X_phases_ref[sensor][phase].keys():
                    X_temp = self._align_arrays(
                        [X_phases_ref[sensor][phase][ind]],
                        max_cols=max_cols[phase],
                        fill_value=0
                    )
                    X_phases_ref_padded[sensor][phase][ind] = X_temp

        return X_phases_ref_padded

    def _get_rel_data_names(self) -> list[str]:
        return ['X', 'y', 'setting', 'y_kg_emb']

    def get_euclidean_data(
            self, max_train_length: int = 900, ood_faults: list[str] = []
    ) -> None:
        self.ood_faults = ood_faults

        # Load data
        self.max_train_length = max_train_length
        X_tr, y_tr, setting_tr = self.load(
            self.path_to_data, 'training',
            self.max_train_length, self.FORCE_RELOAD
        )
        X_v, y_v, setting_v = self.load(
            self.path_to_data, 'validation',
            self.max_train_length, self.FORCE_RELOAD
        )
        X_te, y_te, setting_te = self.load(
            self.path_to_data, 'test',
            self.max_train_length, self.FORCE_RELOAD
        )

        # Remap setting labels for domain classification
        setting_tr = torch.tensor([self.setting_map[s] for s in setting_tr])
        setting_v = torch.tensor([self.setting_map[s] for s in setting_v])
        setting_te = torch.tensor([self.setting_map[s] for s in setting_te])

        # Combine data
        self.sensors = list(X_tr.keys())
        X_tr = torch.stack([torch.tensor(v) for v in X_tr.values()], axis=1)
        X_v = torch.stack([torch.tensor(v) for v in X_v.values()], axis=1)
        X_te = torch.stack([torch.tensor(v) for v in X_te.values()], axis=1)
        X_all = torch.concatenate([X_tr, X_v, X_te], axis=0)
        X_all[torch.isnan(X_all)] = 0
        self.X = X_all
        self.n_sensors = X_all.shape[1]
        self.y = torch.concatenate([
            torch.tensor(v) for v in [y_tr, y_v, y_te]
        ])
        self.n_train_classes = len(self.y.unique()) - len(ood_faults)
        self.y_target_map = {
            self.FAULT_MAPPING[i][0]: i
            for i in range(len(self.y.unique()))
        }
        self.setting = torch.concatenate([
            setting_tr, setting_v, setting_te
        ])

        if len(ood_faults) > 0:
            # Reorder labels to have ood faults at the end
            # So taht
            self._reorder_labels_ood_faults(ood_faults)

    def _get_phases(
            self, split_type: str, individuals: list = None,
            X_ref_po: torch.tensor = None, setting_ref: torch.tensor = None
    ) -> tuple[dict, dict, dict]:
        if split_type == 'phase':
            # Phases based on pressure value
            accel_rear_end, accel_for_start = \
                self._find_acceleration_phases(
                    X_ref_po, setting_ref, individuals
                )
            phase_max_cols = {
                'p1': 400,
                'p2': 100,
                'p3': 400
            }
        elif split_type == 'uniform':
            # Equidistant phases
            phase_len = self.max_train_length // 3
            accel_rear_end = {i: phase_len for i in individuals}
            accel_for_start = {i: phase_len * 2 for i in individuals}
            phase_max_cols = {
                'p1': phase_len,
                'p2': phase_len,
                'p3': phase_len
            }
        elif split_type == 'one':
            # Simple sensor network without phases
            phase_len = self.max_train_length
            phase_max_cols = {'p1': self.max_train_length}
            accel_for_start = None
            accel_rear_end = None
        else:
            raise AttributeError(
                "Invalid phase identification method."
                "Choose either 'pressure', 'equidistant' or 'one'"
                " for phases parameter."
            )
        return accel_rear_end, accel_for_start, phase_max_cols

    def transform_to_graph_data(
            self, edge_attr: bool = True, graph_type: str = 'hetero',
            split_type: str = 'phase', bandwidth: float = 2.0
    ) -> None:
        graph_path = Path(
            f"data/graphs/phm22_rockdrill/{graph_type}"
            f"_edge_{edge_attr}"
            f"_bandwith_{bandwidth}"
            f"_split_{split_type}"
            f"_mtl_{self.max_train_length}.pt"
        )
        euc_data_names = self._get_rel_data_names()
        if (not self.FORCE_RELOAD) and graph_path.exists():
            self.graph_list = self.load_graphs(graph_path)
            self.n_phases = self.graph_list[0]['n_phases']
        else:
            # Get reference data
            idx_ref = self.y == 0
            X_ref_po = self.X[idx_ref, self.sensors.index('po'), :]
            setting_ref = self.setting[idx_ref]
            individuals = [i.item() for i in list(torch.unique(setting_ref))]

            # Get phase indices
            accel_rear_end, accel_for_start, phase_max_cols = self._get_phases(
                split_type, individuals, X_ref_po, setting_ref
            )

            # Split data into phases
            X_phases, y_phases, new_ordering_idx = self._split_data_by_phase(
                self.X, self.y, self.setting, individuals,
                accel_rear_end, accel_for_start
            )

            # Reorder euclidean data to match new ordering
            for data_name in euc_data_names:
                attr = getattr(self, data_name)
                if attr is not None:
                    setattr(self, data_name, attr[new_ordering_idx])

            # Align phase data
            X_phases = self._align_phase_data(X_phases, phase_max_cols)

            # Unpack phase data on setting level and combine
            for s in X_phases.keys():
                for p in X_phases[s].keys():
                    tmp_sig = [
                        v for v in X_phases[s][p].values()
                    ]
                    X_phases[s][p] = torch.concatenate(tmp_sig)

            # Create phase graph for each sample
            self.graph_list = self._create_phase_graph(
                X_phases, y_phases, edge_attr, bandwidth, graph_type,
                additional_info={
                    'setting': self.setting,
                    'y_kg_emb': self.y_kg_emb
                }
            )
            self.n_phases = self.graph_list[0]['n_phases']

            self.save_graphs(self.graph_list, graph_path, new_ordering_idx)
