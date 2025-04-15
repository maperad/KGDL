# RDF-based Knowledge Graph Integration with Deep Learning for Fault Diagnosis

This repository belongs to the paper *RDF-based Knowledge Graph Integration with Deep Learning for Fault Diagnosis*, which is currently under review. Abstract:

> Combining system knowledge with deep learning for fault diagnosis offers the potential to reduce the dependency of deep learning algorithms on extensive labeled datasets.
However, existing methods often rely on highly specialized, problem-specific knowledge or demand detailed physical insights into the system, which limits their generalizability.
Additionally, inconsistencies in knowledge representation hinder the ability to compare and build upon prior approaches.
In this work, we address these challenges by leveraging commonly available knowledge about the phase structure of systems and the hierarchical organization of condition spaces.
This information is systematically represented using \acp{KG} based on the \ac{RDF}.
To integrate this knowledge into deep learning, we transform the input data and the corresponding labels based on the \acp{KG}, and employ a \ac{GNN} trained with a semantic loss function informed by the knowledge about the condition space.
The proposed approach is evaluated on three diverse datasets with varying characteristics under the two scenarios of domain generalization and novel fault detection.

### Install required pacakges
To run the code install all required packages:

````
pip install -r requirements.txt
````

### Download required data
Hydraulic rock drill: https://www.kaggle.com/datasets/erikjakobsson/fault-classification-in-rock-drills

CWRU bearing: https://engineering.case.edu/bearingdatacenter

Spacecraft propulsion: https://phmap.jp/program-data/

Update the ``path_to_data`` options for each dataset in the configuration files in ``configs/``.

### Single run
Set the desired configurations in ``configs/config.yaml`` and run
````
python main.py
````

## Run complete experiments for specific dataset
Basic configurations are set in ``configs/config_experiments``, but most are set directly in ``run_experiments.py``. For the experiments in the paper run
````
python run_experiments.py --dataset PHM22Rockdrill
````
````
python run_experiments.py --dataset PHMAP23Spacecraftpropulsion
````
````
python run_experiments.py --dataset CWRU
````
