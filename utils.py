import os
import pickle
import random

import numpy as np
import torch
import yaml


def ensure_reproduciblity(seed: int = 42) -> None:
    '''
    Sets the seed for various random number generators
    and only use determinisic CUDA algorithms.
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    # When running on the CuDNN backend, further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def yaml_dump(data, file_path: str) -> None:
    with open(file_path, 'w') as file:
        yaml.dump(data, file)


def save_pickle(obj, filename):
    '''
    Save object as pickle.
    '''
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle)


def load_pickle(filename):
    '''
    Load pickled file.
    '''
    with open(filename, 'rb') as handle:
        file = pickle.load(handle)
    return file


def key_value_switch(mapping):
    '''
    Switch key and value in dictionary.
    '''
    return {v: k for k, v in mapping.items()}


def dict_arrays_to_tensors(data_dict):
    '''
    Convert dictionary with arrays to dictionary with tensors.
    '''
    return {
        k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
        for k, v in data_dict.items()
    }
