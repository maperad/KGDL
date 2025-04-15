from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.base import DataPrep
from utils import dict_arrays_to_tensors, key_value_switch


class PHMAP23Spacecraftpropulsion(DataPrep):
    def __init__(
            self, path_to_data: str, force_reload: bool = False,
            kg_name: str = None, seed: int = None
    ):
        super().__init__(force_reload, seed)

        self.kg_name = kg_name.split('/')[-1]
        self.setting_name = 'setting'
        self.spacecraft_path = Path(path_to_data)

        self.NORMAL_MAPPING = {
            0: 'Normal',
            1: 'Abnormal'
        }
        self.CONDITION_MAPPING = {
            0: 'Normal',
            1: 'Unknown',
            2: 'Anomaly',
            3: 'Fault'
        }
        self.BUBBLE_MAPPING = {
            0: 'Normal',
            1: 'BP1',
            2: 'BP2',
            3: 'BP3',
            4: 'BP4',
            5: 'BP5',
            6: 'BP6',
            7: 'BP7',
            8: 'BV1'
        }
        self.SOLENOID_MAPPING = {
            0: 'Normal',
            1: 'SV1',
            2: 'SV2',
            3: 'SV3',
            4: 'SV4'
        }
        self.SOLENOID_SEVERITY_MAPPING = {
            0: '100',
            1: '75',
            2: '50',
            3: '25',
            4: '0'
        }

    def get_new_names(self, sol_sev=False) -> dict[str, str]:
        if sol_sev:
            # Use solenoid severity labels
            return {
                'Anomaly_BP1_Normal_100': 'BP1',
                'Anomaly_BP2_Normal_100': 'BP2',
                'Anomaly_BP3_Normal_100': 'BP3',
                'Anomaly_BP4_Normal_100': 'BP4',
                'Anomaly_BP5_Normal_100': 'BP5',
                'Anomaly_BP6_Normal_100': 'BP6',
                'Anomaly_BP7_Normal_100': 'BP7',
                'Anomaly_BV1_Normal_100': 'BV1',
                'Fault_Normal_SV1_0': 'SV1_0',
                'Fault_Normal_SV1_25': 'SV1_25',
                'Fault_Normal_SV1_50': 'SV1_50',
                'Fault_Normal_SV1_75': 'SV1_75',
                'Fault_Normal_SV2_0': 'SV2_0',
                'Fault_Normal_SV2_25': 'SV2_25',
                'Fault_Normal_SV2_50': 'SV2_50',
                'Fault_Normal_SV2_75': 'SV2_75',
                'Fault_Normal_SV3_0': 'SV3_0',
                'Fault_Normal_SV3_25': 'SV3_25',
                'Fault_Normal_SV3_50': 'SV3_50',
                'Fault_Normal_SV3_75': 'SV3_75',
                'Fault_Normal_SV4_0': 'SV4_0',
                'Fault_Normal_SV4_25': 'SV4_25',
                'Fault_Normal_SV4_50': 'SV4_50',
                'Fault_Normal_SV4_75': 'SV4_75',
                'Normal_Normal_Normal_100': 'Normal',
                'Unknown_Normal_Normal_100': 'Unknown'
            }
        else:
            # Ignore solenoid severity labels
            return {
                'Anomaly_BP1_Normal_100': 'BP1',
                'Anomaly_BP2_Normal_100': 'BP2',
                'Anomaly_BP3_Normal_100': 'BP3',
                'Anomaly_BP4_Normal_100': 'BP4',
                'Anomaly_BP5_Normal_100': 'BP5',
                'Anomaly_BP6_Normal_100': 'BP6',
                'Anomaly_BP7_Normal_100': 'BP7',
                'Anomaly_BV1_Normal_100': 'BV1',
                'Fault_Normal_SV1_0': 'SV1',
                'Fault_Normal_SV1_25': 'SV1',
                'Fault_Normal_SV1_50': 'SV1',
                'Fault_Normal_SV1_75': 'SV1',
                'Fault_Normal_SV2_0': 'SV2',
                'Fault_Normal_SV2_25': 'SV2',
                'Fault_Normal_SV2_50': 'SV2',
                'Fault_Normal_SV2_75': 'SV2',
                'Fault_Normal_SV3_0': 'SV3',
                'Fault_Normal_SV3_25': 'SV3',
                'Fault_Normal_SV3_50': 'SV3',
                'Fault_Normal_SV3_75': 'SV3',
                'Fault_Normal_SV4_0': 'SV4',
                'Fault_Normal_SV4_25': 'SV4',
                'Fault_Normal_SV4_50': 'SV4',
                'Fault_Normal_SV4_75': 'SV4',
                'Normal_Normal_Normal_100': 'Normal',
                'Unknown_Normal_Normal_100': 'Unknown'
            }

    def _label_col_order(self, labels):
        return labels[[
            'CaseNo', 'SpacecraftNo', 'Condition', 'ConditionNum',
            'BubbleCondition', 'BubbleConditionNum', 'SolenoidFaultLocation',
            'SolenoidFaultLocationNum', 'NormalAbnormal', 'NormalAbnormalNum',
            'SolenoidFaultSeverityClf', 'SolenoidFaultSeverityClfNum',
            'SolenoidFaultSeverityReg'
        ]]

    def _load_train_data(self, path_to_data):
        # Load sensor data
        train_files = (path_to_data / 'train' / 'data').rglob('*.csv')
        train_data = {
            f.stem: pd.read_csv(f) for f in train_files
        }
        # Load and process labels
        train_labels = pd.read_excel(
            path_to_data / 'train/labels.xlsx', skiprows=1
        )
        names = ['CaseNo', 'SpacecraftNo', 'Condition']
        train_labels.rename(
            columns={train_labels.columns[i]: names[i] for i in range(3)},
            inplace=True
        )
        train_labels['ConditionNum'] = train_labels.Condition.map(
            key_value_switch(self.CONDITION_MAPPING)
        )
        train_labels['BubbleCondition'] = 'Normal'
        train_labels['SolenoidFaultLocation'] = 'Normal'
        train_labels['SolenoidFaultSeverityClf'] = '100'
        train_labels['SolenoidFaultSeverityReg'] = 100
        for i in range(len(train_labels)):
            for col in ['SV1', 'SV2', 'SV3', 'SV4']:
                if train_labels[col][i] != 100:
                    train_labels.at[i, 'SolenoidFaultLocation'] = col
                    train_labels.at[i, 'SolenoidFaultSeverityReg'] = \
                        train_labels[col][i]
                    train_labels.at[i, 'SolenoidFaultSeverityClf'] = str(
                        train_labels[col][i]
                    )
            for col in [
                'BP1', 'BP2', 'BP3', 'BP4', 'BP5', 'BP6', 'BP7', 'BV1'
            ]:
                if train_labels[col][i] == 'Yes':
                    train_labels.at[i, 'BubbleCondition'] = col
        train_labels.drop(columns=[
            'SV1', 'SV2', 'SV3', 'SV4',
            'BP1', 'BP2', 'BP3', 'BP4', 'BP5', 'BP6', 'BP7', 'BV1'
        ], inplace=True)
        train_labels['NormalAbnormal'] = 'Abnormal'
        train_labels.loc[
            train_labels.Condition == 'Normal', 'NormalAbnormal'
            ] = 'Normal'
        train_labels['BubbleConditionNum'] = train_labels[
            'BubbleCondition'
        ].map(
            key_value_switch(self.BUBBLE_MAPPING)
        )
        train_labels['SolenoidFaultLocationNum'] = train_labels[
            'SolenoidFaultLocation'
        ].map(
            key_value_switch(self.SOLENOID_MAPPING)
        )
        train_labels['NormalAbnormalNum'] = train_labels[
            'NormalAbnormal'
        ].map(
            key_value_switch(self.NORMAL_MAPPING)
        )
        train_labels['SolenoidFaultSeverityClfNum'] = train_labels[
            'SolenoidFaultSeverityClf'
        ].map(
            key_value_switch(self.SOLENOID_SEVERITY_MAPPING)
        )
        train_labels = self._label_col_order(train_labels)
        return train_data, train_labels

    def _load_test_data(self, data_path):
        # Load sensor data
        test_files = (data_path / 'test' / 'data').rglob('*.csv')
        test_data = {
            f.stem: pd.read_csv(f) for f in test_files
        }
        # Load and process labels
        test_labels = pd.read_csv(data_path / 'test/answer.csv')

        test_labels.rename(
            columns={'task1': 'NormalAbnormalNum'}, inplace=True
        )

        test_labels['NormalAbnormal'] = test_labels[
            'NormalAbnormalNum'
        ].map(self.NORMAL_MAPPING)
        test_labels.rename(columns={'task2': 'ConditionNum'}, inplace=True)

        test_labels['Condition'] = test_labels.ConditionNum.map(
            self.CONDITION_MAPPING
        )

        test_labels.rename(
            columns={'task3': 'BubbleConditionNum'}, inplace=True
        )

        test_labels['BubbleCondition'] = test_labels[
            'BubbleConditionNum'
        ].map(self.BUBBLE_MAPPING)

        test_labels.rename(
            columns={'task4': 'SolenoidFaultLocationNum'}, inplace=True
        )

        test_labels['SolenoidFaultLocation'] = test_labels[
            'SolenoidFaultLocationNum'
        ].map(self.SOLENOID_MAPPING)

        test_labels['SolenoidFaultSeverityClf'] = ''
        for i in range(len(test_labels)):
            if test_labels.task5[i] == 100:
                test_labels.at[i, 'SolenoidFaultSeverityClf'] = '100'
            elif (test_labels.task5[i] < 100) & (test_labels.task5[i] >= 75):
                test_labels.at[i, 'SolenoidFaultSeverityClf'] = '75'
            elif (test_labels.task5[i] < 75) & (test_labels.task5[i] >= 50):
                test_labels.at[i, 'SolenoidFaultSeverityClf'] = '50'
            elif (test_labels.task5[i] < 50) & (test_labels.task5[i] >= 25):
                test_labels.at[i, 'SolenoidFaultSeverityClf'] = '25'
            else:
                test_labels.at[i, 'SolenoidFaultSeverityClf'] = '0'
        test_labels['SolenoidFaultSeverityClfNum'] = test_labels[
            'SolenoidFaultSeverityClf'
        ].map(key_value_switch(self.SOLENOID_SEVERITY_MAPPING))
        test_labels.rename(
            columns={'task5': 'SolenoidFaultSeverityReg'}, inplace=True
        )
        test_labels.drop(columns=['Test condition'], inplace=True)
        test_labels.rename(
            columns={
                'Spacecraft No.': 'SpacecraftNo',
                'ID': 'CaseNo'
            }, inplace=True
        )
        test_labels = self._label_col_order(test_labels)
        return test_data, test_labels

    def _extract_arrays(self, data_dict, labels):
        X = []
        t = []
        space_craft_no = []
        y_normal = []
        y_condition = []
        y_bubble = []
        y_solenoid_loc = []
        y_solenoid_sev_clf = []
        y_solenoid_sev_reg = []
        data_arrays = {}
        for case_no in labels.CaseNo:
            case_labels = labels.loc[labels.CaseNo == case_no]
            space_craft_no.append(case_labels['SpacecraftNo'].values[0])
            X.append(data_dict[f'Case{case_no:03d}'].iloc[:, 1:].values)
            t.append(data_dict[f'Case{case_no:03d}'].iloc[:, 0].values)
            y_normal.append(case_labels['NormalAbnormalNum'].values[0])
            y_condition.append(case_labels['ConditionNum'].values[0])
            y_bubble.append(case_labels['BubbleConditionNum'].values[0])
            y_solenoid_loc.append(
                case_labels['SolenoidFaultLocationNum'].values[0]
            )
            y_solenoid_sev_clf.append(
                case_labels['SolenoidFaultSeverityClfNum'].values[0]
            )
            y_solenoid_sev_reg.append(
                case_labels['SolenoidFaultSeverityReg'].values[0]
            )
        data_arrays['space_craft_no'] = np.array(space_craft_no)
        data_arrays['X'] = np.stack(X)
        data_arrays['t'] = np.stack(t)
        data_arrays['y_normal'] = np.array(y_normal)
        data_arrays['y_condition'] = np.array(y_condition)
        data_arrays['y_bubble'] = np.array(y_bubble)
        data_arrays['y_solenoid_loc'] = np.array(y_solenoid_loc)
        data_arrays['y_solenoid_sev_clf'] = np.array(y_solenoid_sev_clf)
        data_arrays['y_solenoid_sev_reg'] = np.array(y_solenoid_sev_reg)
        return data_arrays

    def _combine_labels(
            self, y_condition, y_bubble, y_solenoid_loc, y_solenoid_sev_clf
    ):
        mapped_y_condition = [
            self.CONDITION_MAPPING.get(int(i)) for i in y_condition
        ]
        mapped_y_bubble = [self.BUBBLE_MAPPING.get(int(i)) for i in y_bubble]
        mapped_y_solenoid_loc = [
            self.SOLENOID_MAPPING.get(int(i)) for i in y_solenoid_loc
        ]
        mapped_y_solenoid_sev_clf = [
            self.SOLENOID_SEVERITY_MAPPING.get(int(i))
            for i in y_solenoid_sev_clf
        ]
        mapped_y = [
            f'{mapped_y_condition[i]}_{mapped_y_bubble[i]}_'
            f'{mapped_y_solenoid_loc[i]}_{mapped_y_solenoid_sev_clf[i]}'
            for i in range(len(y_condition))
        ]
        y_mapping = {
            k: i for i, k in enumerate(sorted(set(mapped_y)))
        }
        y = np.array([y_mapping[k] for k in mapped_y])
        return y, y_mapping

    def load(self, data_path, train=True, test=True, tensor=False):
        arrays = {}
        if train:
            train_data, train_labels = self._load_train_data(data_path)
            arrays['train'] = self._extract_arrays(train_data, train_labels)
        if test:
            test_data, test_labels = self._load_test_data(data_path)
            arrays['test'] = self._extract_arrays(test_data, test_labels)
        if train and test:
            # Combine train and test data
            arrays = {
                k: np.concatenate([arrays['train'][k], arrays['test'][k]])
                for k in arrays['train'].keys()
            }
        else:
            arrays = arrays['train'] if train else arrays['test']
        y, y_mapping = self._combine_labels(
            arrays['y_condition'],
            arrays['y_bubble'],
            arrays['y_solenoid_loc'],
            arrays['y_solenoid_sev_clf']
        )
        arrays['y'] = y
        arrays['y_mapping'] = y_mapping
        if tensor:
            arrays = dict_arrays_to_tensors(arrays)
        return arrays

    def get_euclidean_data(
            self,
            split: bool = False,
            label_solenoid_sev: bool = False,
            include_unknown: bool = False,
            ood_faults: list[str] = []
    ) -> None:

        data_dict = self.load(
            data_path=self.spacecraft_path, tensor=True
        )
        self.X = torch.permute(data_dict['X'], (0, 2, 1))
        self.y = data_dict['y']
        self.ood_faults = ood_faults
        self.n_sensors = self.X.shape[1]
        self.sensors = [f's{i+1}' for i in range(self.n_sensors)]
        self.y_target_map = data_dict['y_mapping']
        self.setting = data_dict['space_craft_no']
        self.split = split

        if not include_unknown:
            # Delete unknown samples (hard to evaluate)
            not_unknown_idx = torch.where(
                self.y != self.y_target_map['Unknown_Normal_Normal_100']
            )[0]
            self.X = self.X[not_unknown_idx, :, :]
            self.y = self.y[not_unknown_idx]
            self.setting = self.setting[not_unknown_idx]
            self.y_target_map.pop('Unknown_Normal_Normal_100')

        # Map labels to new names for kg embeddings
        new_names = self.get_new_names(label_solenoid_sev)
        new_label_map = {
            v: i for i, v in enumerate(sorted(set(new_names.values())))
        }
        label_to_name = {v: k for k, v in self.y_target_map.items()}
        self.y = torch.tensor([
            new_label_map[new_names[label_to_name[int(v)]]] for v in self.y
        ])
        self.y_target_map = new_label_map

        if not include_unknown:
            for k in list(self.y_target_map.keys()):
                if k.startswith('Unknown'):
                    self.y_target_map.pop(k, None)

        if len(ood_faults) > 0:
            # Reorder labels to have ood faults at the end
            # So taht
            self._reorder_labels_ood_faults(ood_faults)

        if self.split:
            # Split samples into 3 sub-samples
            # (one sample for one open close cycle)
            open_close_time = 400  # ms
            n_splits = self.X.shape[2] // open_close_time
            X, y, setting = [], [], []
            for i in range(n_splits):
                X.append(self.X[
                    :, :, i*open_close_time:(i+1)*open_close_time
                ])
                y.append(self.y)
                setting.append(self.setting)
            self.X = torch.cat(X)
            self.y = torch.cat(y)
            self.setting = torch.cat(setting)

        self.n_train_classes = len(self.y.unique()) - len(ood_faults)

    def _get_phases(self, split_type: str = 'phase') -> tuple:
        if split_type == 'phase':
            # Open and close times of the solenoid valves
            # Opens after 100 ms and closes after another 300 ms
            open_times = torch.tensor([100, 500, 900])
            close_times = torch.tensor([400, 800, 1200])
            comb_times = torch.tensor([0, 100, 400, 500, 800, 900, 1200])
        elif split_type == 'uniform':
            # Split uniformly into 3 phases
            open_times = None
            close_times = None
            comb_times = torch.linspace(0, 1200, 7).int()
        elif split_type == 'one':
            open_times = None
            close_times = None
            comb_times = None
        return open_times, close_times, comb_times

    def _get_rel_data_names(self) -> list[str]:
        return ['X', 'y', 'setting', 'y_kg_emb']

    def _split_data_by_phase(
            self, X: torch.Tensor, comb_times: torch.Tensor
    ) -> dict[str, dict[str, torch.Tensor]]:
        if comb_times is None:
            # Only one phase (no split)
            return {
                s: {'p1': X[:, i, :]} for i, s in enumerate(self.sensors)
            }
        # Split data into phases according to standard open and close times
        if self.split:
            # Only one open close cycle
            phases = ['p1', 'p2']
        else:
            # 3 open close cycles
            phases = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']
        X_phases = {
            s: {
                p: [] for p in phases
            } for s in self.sensors
        }
        for i, s in enumerate(self.sensors):
            for j, p in enumerate(phases):
                X_phases[s][p] = X[
                    :, i, comb_times[j]:comb_times[j+1]
                ]

        return X_phases

    def transform_to_graph_data(
            self, edge_attr: bool = True, graph_type: str = 'hetero',
            split_type: str = 'phase'
    ) -> None:
        graph_path = Path(
            f"data/graphs/phmap23_spacecraftpropulstion/{graph_type}"
            f"_edge_{edge_attr}"
            f"_split_{split_type}"
            f"_split_{self.split}.pt"
        )
        if (not self.FORCE_RELOAD) and graph_path.exists():
            self.graph_list = self.load_graphs(graph_path)
            self.n_phases = self.graph_list[0]['n_phases']
        else:
            # Get phase indices
            _, _, comb_times = self._get_phases(split_type)

            # Split data into phases
            X_phases = self._split_data_by_phase(self.X, comb_times)

            # Create phase graph for each sample
            self.graph_list = self._create_phase_graph(
                X_phases, self.y, edge_attr, graph_type,
                additional_info={
                    'setting': self.setting,
                    'y_kg_emb': self.y_kg_emb
                }
            )
            self.n_phases = self.graph_list[0]['n_phases']

            self.save_graphs(self.graph_list, graph_path)
