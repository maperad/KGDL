# %%
from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt


def flatten_dict(
        d: dict, parent_key: str = '', sep: str = '_',
        flatten_lists: bool = False
) -> dict:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(
                v, new_key, sep=sep, flatten_lists=flatten_lists
            ).items())
        elif isinstance(v, list):
            if flatten_lists:
                for i, item in enumerate(v):
                    items.append((f"{new_key}_{i}", item))
            else:
                items.append((new_key, v))
        else:
            items.append((new_key, v))
    return dict(items)


# %%
dataset = "PHMAP23Spacecraftpropulsion"
experiment_name = "2025-10-14_sensitivity_temperature_bandwidth"

# %%
experiment_path = Path(
    "/home/mpradtke/01_DEV/PhaseGraph_KGLoss/experiments/"
)
experiment_path = experiment_path / f"{dataset}/{experiment_name}"

# Get all config and metric paths of experiment
config_paths = list(experiment_path.glob("*/config.yaml"))
metrics_paths = list(experiment_path.glob("*/metrics.yaml"))

# Load config and metric files
config_files = {}
metric_files = {}
i = 0
for path in config_paths:
    with open(path) as file:
        config_files[i] = yaml.load(file, Loader=yaml.FullLoader)
    i += 1
i = 0
for path in metrics_paths:
    with open(path) as file:
        metric_files[i] = yaml.load(file, Loader=yaml.FullLoader)
    i += 1

# Save all configs and metrics in one dataframe
datasets = ['CWRU', 'PHM22Rockdrill', 'PHMAP23Spacecraftpropulsion']
df_out = pd.DataFrame()
for i, config in config_files.items():
    metric = metric_files[i]

    # Delete irrelevant dataset inforamtion
    rel_dataset = config['dataset']
    for k_data in list(config['data'].keys()):
        if (k_data in datasets) and (k_data != rel_dataset):
            config['data'].pop(k_data)

    config = flatten_dict(config)
    config_cols = list(config.keys())
    config_values = [str(v) for v in config.values()]
    metric_cols = list(metric.keys())
    metric_values = list(metric.values())
    df_config = pd.DataFrame([config_values], columns=config_cols)
    df_metric = pd.DataFrame([metric_values], columns=metric_cols)
    df_i = pd.concat([df_config, df_metric], axis=1)
    df_out = pd.concat([df_out, df_i], axis=0)
# df_out.drop('model_summary', axis=1, inplace=True)
# df_out.drop('seed', axis=1, inplace=True)
# df_out.drop('cuda_device', axis=1, inplace=True)
metric_cols = [
    col for col in df_out.columns
    if (
        (('loss' in col) or ('clf' in col)
            or ('kg_cent' in col) or ('acc' in col)
            or ('recall' in col) or ('precision' in col)
            or ('f1' in col) or ('mrr' in col)
            or ('_dist_' in col))
        and (not col.startswith('loss_'))
    )
]
# config_cols = [col for col in df_out.columns if col not in metric_cols]

# # Delete columns that are not relevant for all metrics
# metric_cols = [
#     col
#     for col in metric_cols
#     if col not in ['emb_dist_thr_class', 'proj_dist_thr_class']
# ]

# Add train settings
tr_set_col = f"data_{rel_dataset}_train_settings"
df_out['n_tr_settings'] = df_out[
    tr_set_col
].apply(lambda x: len(x.split(', ')))

if rel_dataset == 'PHMAP23Spacecraftpropulsion':
    # Exclude rows where setting 4 is tested on
    df_out = df_out[df_out[tr_set_col].apply(lambda x: '4' in x)]

# Add ood flag
ood_col = f"data_{dataset}_load_ood_faults"
df_out['ood'] = df_out[ood_col].apply(
    lambda x: True if x != "[]" else False
)

# %%
ood_faults = '[]'
ood_col = f"data_{dataset}_load_ood_faults"
train_settings = '(3, 4)'
tr_set_col = f"data_{dataset}_train_settings"
df_plot = df_out.loc[
    (df_out[ood_col] == ood_faults)
    & (df_out[tr_set_col] == train_settings)
]
# %%
bw_col = 'model_GNN_graph_bandwidth'
df_plot[bw_col] = df_plot[bw_col].astype(float)
temp_col = 'loss_embedding_temperature'
df_plot[temp_col] = df_plot[temp_col].astype(float)


# %%
df_bw = df_plot.loc[df_plot[temp_col] == 0.1, ]
df_bw_mean = df_bw[[bw_col, 'clf_test_setting_acc']].groupby(bw_col).mean().reset_index()
df_bw_std = df_bw[[bw_col, 'clf_test_setting_acc']].groupby(bw_col).std().reset_index()
# %%
df_tmp = df_plot.loc[df_plot[bw_col] == 1, ]
df_tmp_mean = df_tmp[[temp_col, 'clf_test_setting_acc']].groupby(temp_col).mean().reset_index()
df_tmp_std = df_tmp[[temp_col, 'clf_test_setting_acc']].groupby(temp_col).std().reset_index()
# %%