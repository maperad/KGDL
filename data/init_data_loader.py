from data.cwru import CWRU
from data.phm22_rockdrill import PHM22Rockdrill
from data.phmap23_spacecraftpropulsion import PHMAP23Spacecraftpropulsion


def init_data_loader(
        dataset: str, path_to_data: str, force_reload: bool = False,
        kg_name: str = None, seed: int = None
):
    '''
    Select and initialize the data loader based on the dataset.

    Args:
        dataset (str): The dataset to load.
        path_to_data (str): Path to the dataset.
        force_reload (bool): Whether to force reload the data.
        kg_name (str): The name of the knowledge graph.
        seed (int): Random seed for reproducibility.

    Returns:
        DataLoad: The data loader.
    '''
    available_datasets = ['CWRU', 'PHM22Rockdrill']
    if dataset == 'CWRU':
        return CWRU(path_to_data, force_reload, kg_name, seed)
    elif dataset == 'PHM22Rockdrill':
        return PHM22Rockdrill(path_to_data, force_reload, kg_name, seed)
    elif dataset == 'PHMAP23Spacecraftpropulsion':
        return PHMAP23Spacecraftpropulsion(
            path_to_data, force_reload, kg_name, seed
        )
    else:
        raise ValueError(
            f"Dataset {dataset} not found."
            f"Please select one of the following: {available_datasets}"
        )
