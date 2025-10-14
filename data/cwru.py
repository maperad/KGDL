import torch
import numpy as np
from pathlib import Path
import logging
import re

from scipy.signal import decimate
from scipy.io import loadmat

from utils import save_pickle, load_pickle

from data.base import DataPrep


class CWRU(DataPrep):
    def __init__(
            self, path_to_data: str, force_reload: bool = False,
            kg_name: str = None, seed: int = None
    ):
        super().__init__(force_reload, seed)

        self.kg_name = kg_name.split('/')[-1]
        self.setting_name = 'rpm'
        self.path_to_data = Path(path_to_data)
        self.rpm = None
        self.y_target_map = None
        self.force_reload = force_reload

        self.rpm_mapping = {
            '0': 1797,
            '1': 1772,
            '2': 1750,
            '3': 1730
        }

    def _cwru_load_raw_data(self, path, measure_type, force_reload=False):
        dom = path.name
        file_name = f'cwru_{dom}_{measure_type}_data.pkl'
        if (path / file_name).is_file() and not force_reload:
            logging.info('Load %s.', file_name)
            data = load_pickle(path / file_name)
            return data
        logging.info(f'Load CWRU {dom} data from source.')
        all_files = path.glob('*.mat')
        data = {}
        for i in all_files:
            name = i.name
            tmp_data = loadmat(i)
            data[name] = {
                k.split('_')[1]: v.squeeze()
                for k, v in tmp_data.items() if k.endswith(measure_type)
            }
        save_pickle(data, path / file_name)
        return data

    def _raw_to_segmented(self, data, segment_length, sample_rate=12):
        # Initialize results dictionary
        dic_cwru = {
            'PositionSize': [],
            'RPM': [],
            'Signal': []
        }

        # Fill results dictionary
        for f, signal in data.items():
            is_normal = f.startswith('Normal')
            signal = signal['DE']

            # Get labels
            if is_normal:
                pos_size = 'Normal'
            else:
                size = re.search(r"([0-9]{3})", f)[1]
                if size == '028':
                    # Exclude 028 bearing faults
                    continue
                pos = re.search(r'([A-Z]*)', f)[1]
                pos_size = f"{size}{pos}"
            rpm = re.search(r"(\d+)(?!.*\d)", f)[1]

            # Get signal segments
            if is_normal:
                if 48 != sample_rate:
                    down_sample = 48 // sample_rate
                    signal = decimate(signal, q=down_sample)
            n_sample_points = len(signal)
            n_segments = n_sample_points // segment_length
            segment_list = []
            for segment in range(n_segments):
                segment_data = signal[
                    segment_length * segment:segment_length*(segment+1)
                ]
                segment_list.append(segment_data)
            dic_cwru["Signal"].append(np.array(segment_list))

            # Add labels
            len_segments = len(segment_list)
            dic_cwru['PositionSize'] += [pos_size] * len_segments
            dic_cwru['RPM'] += [rpm] * len_segments

        # Transform lists to arrays
        for k, v in dic_cwru.items():
            if k == 'Signal':
                dic_cwru[k] = np.concatenate(v, axis=0)
            else:
                dic_cwru[k] = np.array(v)

        return dic_cwru

    def _load_cwru_dict(
            self, sample_rate: int = 12, segment_length: int = 256
    ) -> dict:
        data_fault = self._cwru_load_raw_data(
            self.path_to_data / '12k_DE',
            measure_type='DE_time',
            force_reload=self.force_reload
        )
        data_normal = self._cwru_load_raw_data(
            self.path_to_data / 'Normal',
            measure_type='DE_time',
            force_reload=self.force_reload
        )
        data = {**data_fault, **data_normal}
        dic_cwru = data
        dic_cwru = self._raw_to_segmented(data, segment_length, sample_rate)
        dic_cwru['RPM'] = np.vectorize(
            self.rpm_mapping.get
        )(dic_cwru['RPM'])

        return dic_cwru

    def get_euclidean_data(
            self, sample_rate: int, segment_length: int,
            ood_faults: list[str] = []
    ) -> None:
        self.n_sensors = 1
        self.ood_faults = ood_faults
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.obs_time = segment_length / sample_rate * 1000
        data_path = Path(
            f"data/euclidean/cwru/{sample_rate}"
            f"_{segment_length}"
            f"_ood_{self.ood_faults}.pt"
        )
        if (not self.FORCE_RELOAD) and data_path.exists():
            data_dict = self.load_euclidean_data(data_path)
            self.X = data_dict['X']
            self.y = data_dict['y']
            self.rpm = data_dict['rpm']
            self.y_target_map = data_dict['y_target_map']
        else:
            cwru_dict = self._load_cwru_dict(
                sample_rate, segment_length
            )
            self.y_target_map = self._y_str_to_int_label_mapping(
                cwru_dict['PositionSize']
            )
            self.X = torch.tensor(
                cwru_dict['Signal'], dtype=torch.float32
            )[:, None, :]
            self.y = self._y_str_to_int(
                cwru_dict['PositionSize'], self.y_target_map
            )
            self.rpm = torch.tensor(cwru_dict['RPM'], dtype=torch.long)
            if len(ood_faults) > 0:
                # Reorder labels to have ood faults at the end
                self._reorder_labels_ood_faults(ood_faults)
            self.save_euclidean_data(
                {'X': self.X, 'y': self.y, 'rpm': self.rpm,
                 'y_target_map': self.y_target_map},
                data_path
            )
        self.n_train_classes = len(self.y.unique()) - len(ood_faults)

    def _get_rel_data_names(self) -> list[str]:
        return ['X', 'rpm', 'y', 'y_kg_emb']

    def _split_data_by_phase(
            self,
            phase_len: dict,
            train_phase_len: int,
            split_type: str = 'phase'
    ) -> dict:
        if split_type == 'one':
            self.n_phases = 1
            return {
                '1': {'p1': self.X.squeeze()}
            }

        self.n_phases = self.X.shape[1] // train_phase_len + 1
        phases = [f'p{i+1}' for i in range(self.n_phases)]
        X_s_p = {
            '1': {
                p: torch.zeros((len(self.y), train_phase_len))
                for p in phases
                }
        }
        phase = {
            p: [] for p in phases
        }
        if split_type == 'phase':
            for r in self.rpm.unique():
                idx = torch.where(self.rpm == r)[0]
                p_len = phase_len[r.item()]
                for j, p in enumerate(phase.keys()):
                    start = j*p_len
                    end = (j+1)*p_len
                    p_sig = self.X[idx, start:end].squeeze()
                    if p_sig.shape[1] < train_phase_len:
                        X_s_p['1'][p][idx, :p_sig.shape[1]] = p_sig
                    else:
                        # Cut of the end of the signal
                        X_s_p['1'][p][idx] = p_sig[:, :train_phase_len]

        return X_s_p

    def transform_to_graph_data(
            self, edge_attr: bool = True, graph_type: str = 'hetero',
            split_type: str = 'phase', bandwidth: float = 2.0
    ) -> None:
        graph_path = Path(
            f"data/graphs/cwru/{self.sample_rate}"
            f"_{self.segment_length}"
            f"_split_{split_type}"
            f"_bandwith_{bandwidth}"
            f"_{graph_type}_edge_{edge_attr}.pt"
        )
        if (not self.FORCE_RELOAD) and graph_path.exists():
            self.graph_list = self.load_graphs(graph_path)
            self.n_phases = self.graph_list[0]['n_phases']
        else:
            time = np.arange(self.segment_length) / (self.sample_rate * 1000)
            phase_len = {
                r: int(sum(time < 1/(r/60)))
                for r in np.unique(self.rpm)
            }  # one rotation represents one phase

            max_phase_len = 417

            X_s_p = self._split_data_by_phase(
                phase_len, max_phase_len, split_type=split_type
            )

            self.graph_list = self._create_phase_graph(
                X_s_p, self.y, edge_attr, bandwidth, graph_type,
                {'y_score': self.y_score,
                 'y_kg_emb': self.y_kg_emb,
                 'rpm': self.rpm,
                 'X': self.X}
            )

            self.save_graphs(self.graph_list, graph_path)
