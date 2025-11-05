import argparse
import copy
import itertools
from datetime import datetime

import yaml
from tqdm import tqdm

from main import main


def run_sensitivity_analysis(
        dataset, exp_name, test=False, save_models=False, cuda_device=0
):

    # Load config files #######################################################

    with open('configs/config_experiments.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    experiment_name = (f"{datetime.now().strftime('%Y-%m-%d')}_{exp_name}")
    config['experiment_name'] = experiment_name
    config['cuda_device'] = cuda_device

    # TODO adjust for final push
    config['data']['CWRU']['path_to_data'] = \
        '/home/mpradtke/00_DATA/CWRU_Bearing_Diagnostic/Data'
    config['data']['PHM22Rockdrill']['path_to_data'] = \
        '/home/mpradtke/00_DATA/PHM22_RockDrill_Diagnostic'
    config['data']['PHMAP23Spacecraftpropulsion']['path_to_data'] = \
        '/home/mpradtke/00_DATA/PHMAP23_SpacecraftPropulsion_Diagnostic'

    ###########################################################################

    # General settings (Same for all datasets) ################################

    config['dataset'] = dataset
    random_seeds = [67,  87, 660,  71, 695, 950, 516, 685, 641, 320]
    if test:
        random_seeds = [1, 2]
    config['data']['train_size'] = 0.8

    ###########################################################################

    # Dataset specific settings ###############################################

    # Batch size
    batch_size = {
        'CWRU': 128, 'PHM22Rockdrill': 128, 'PHMAP23Spacecraftpropulsion': 16
    }

    # Training epochs
    epochs_clf = 200
    epoch_emb = {
        'CWRU': 200, 'PHM22Rockdrill': 100, 'PHMAP23Spacecraftpropulsion': 200
    }
    if test:
        # Test script with less epochs
        epochs_clf = 2
        epoch_emb = {
            'CWRU': 2, 'PHM22Rockdrill': 2, 'PHMAP23Spacecraftpropulsion': 2
        }

    # Training settings
    train_settings = {
        'CWRU': [], 'PHM22Rockdrill': [], 'PHMAP23Spacecraftpropulsion': []
    }
    # for i in [2, 3, 4]:
    #     train_settings['CWRU'].extend(
    #         list(itertools.combinations(['0', '1', '2', '3'], i))
    #     )
    # for i in [2, 3, 4]:
    #     # Ensure 4 is always in the train settings
    #     train_settings['PHMAP23Spacecraftpropulsion'].extend(
    #         [
    #             comb for comb in itertools.combinations([1, 2, 3, 4], i)
    #             if 4 in comb
    #         ]
    #     )
    train_settings['PHMAP23Spacecraftpropulsion'] = [[2, 3, 4]]
    # train_settings['PHM22Rockdrill'] = [
    #     [0, 1, 2, 3], [4, 5, 6, 7],
    #     [0, 1, 2, 3, 4], [3, 4, 5, 6, 7],
    #     [0, 1, 2, 3, 4, 5], [2, 3, 4, 5, 6, 7]
    # ]

    # Out of distribution (OOD) faults
    ood_fault_lists = {
        'CWRU': [
            [],
            # ['007IR', '014IR', '021IR'],
            # ['007OR', '014OR', '021OR'],
            # ['007B', '014B', '021B']
        ],
        'PHM22Rockdrill': [
            [],
            # ['R', 'C'],
            # ['A', 'B', 'O'],
            # ['D', 'Q'],
            # ['V'],
            # ['S', 'T']
        ],
        'PHMAP23Spacecraftpropulsion': [
            [],
            # ['BV1', 'BP1', 'BP2', 'BP3', 'BP4', 'BP5', 'BP6', 'BP7'],
            # ['SV1', 'SV2', 'SV3', 'SV4']
        ]
    }

    # KG path for KG Loss
    kg_dir = 'kgs/'
    kg_paths = {
        'CWRU': kg_dir + 'cwru_fault_hierarchy.ttl',
        'PHM22Rockdrill': kg_dir + 'phm22_rockdrill_fault_hierarchy.ttl',
        'PHMAP23Spacecraftpropulsion':
            kg_dir + 'phmap23_spacecraftpropulsion_fault_hierarchy.ttl'
    }

    # Set data options
    config['data']['batch_size'] = batch_size[dataset]
    train_settings_lists = train_settings[dataset]
    ood_fault_lists = ood_fault_lists[dataset]
    kg_path = kg_paths[dataset]
    data_options = [
        (seed, train_setting, ood_faults, kg_path)
        for seed in random_seeds
        for train_setting in train_settings_lists
        for ood_faults in ood_fault_lists
    ]

    ###########################################################################

    # Model specific settings (Same for all datasets) #########################
    config['loss'] = {
        'clf': {'epochs': epochs_clf},
        'embedding': {
             'epochs': epoch_emb[dataset],
             'name': 'kg',
             'temperature': None
        }
    }

    config['model'] = {
        'type': 'GNN',
        'GNN':  {
            'graph': {
                'split_type': 'phase',
                'edge_attr': True
            },
            'cnn_pre': True
        }
    }

    ###########################################################################

    # Hyperparameters for sensitvity analysis #################################
    temp_val_options = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14]
    bandwidth_options = [0.1, 0.2, 0.5, 1, 2, 5]

    hyperparameters = [
        (temp, 2)  # Default bandwidth 2
        for temp in temp_val_options
    ]
    hyperparameters += [
        (0.1, bandwidth)  # Default temperature 0.1
        for bandwidth in bandwidth_options
    ]

    ###########################################################################

    # Run sensitivity analysis ################################################
    data_config = config['data'][dataset]

    p_bar = tqdm(data_options, desc='Experiments')

    for seed, train_setting, ood_faults, kg_path in p_bar:
        config['seed'] = seed
        data_config['train_settings'] = train_setting
        data_config['load']['ood_faults'] = ood_faults
        data_config['kg_path'] = kg_path
        config['data'][dataset] = data_config

        hyperparameter_bar = tqdm(
            hyperparameters, leave=False, desc='Hyperparameters'
        )
        for temp, bandwidth in hyperparameter_bar:
            single_run_config = copy.deepcopy(config)
            single_run_config['loss']['embedding']['temperature'] = temp
            single_run_config['model']['GNN']['graph']['bandwidth'] = bandwidth
            main(single_run_config, save_models)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, default='PHMAP23Spacecraftpropulsion'
    )
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--save_models', action="store_true")
    parser.add_argument('--cuda_device', type=int, default=0)
    args = parser.parse_args()
    run_sensitivity_analysis(
        args.dataset, args.exp_name, args.test, args.save_models,
        args.cuda_device
    )
