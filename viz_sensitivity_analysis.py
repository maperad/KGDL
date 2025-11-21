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


def set_size(width, fraction=1):
    """
    https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = [fig_width_in, fig_height_in]

    return fig_dim


def set_plot_style(ds='CWRU'):
    # Set plot style
    plt.style.use('tableau-colorblind10')
    text_width = 384
    fig_dim = set_size(text_width)
    # fig_dim[1] *= 1.3

    colors = [
        '#003a7d',
        '#ff73b6',
        '#4ecb8d',
        '#ff9d3a',
        '#008dff',
        '#d83034',
        '#c701ff',
        '#f9e858',
        '#595959',
        '#d4d4d4'
    ]
    if ds == 'CWRU':
        target_y_map_pub_map = {
            '021IR': 'IR21',
            '007IR': 'IR07',
            '014IR': 'IR14',
            '014OR': 'OR14',
            '007OR': 'OR07',
            '021OR': 'OR21',
            '014B': 'B14',
            '007B': 'B07',
            '021B': 'B21',
            'Normal': 'noFault'
        }
    else:
        target_y_map_pub_map = {}

    label_size = 10

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": label_size,  # 8,
        "font.size": label_size,  # 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": label_size-2,  # 6,
        "xtick.labelsize": label_size-2,  # 6,
        "ytick.labelsize": label_size-2,  # 6
    }
    # Latex needs to be installed on the system
    plt.rcParams.update(tex_fonts)
    return fig_dim, colors, target_y_map_pub_map

fig_dim, colors, _ = set_plot_style()

# %%
dataset = "PHMAP23Spacecraftpropulsion"
# experiment_name = "2025-11-05_sens_analysis_234_noood"
# experiment_name = "2025-11-11_sens_analysis_234"
experiment_name = "2025-11-11_sens_analysis_134"
train_settings = '[2, 3, 4]'

# dataset = 'PHM22Rockdrill'
# experiment_name = '2025-11-05_sens_analysis_0123_noood'
# train_settings = '[0, 1, 2, 3]'
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
# ood_faults = '[]'
# ood_col = f"data_{dataset}_load_ood_faults"
# tr_set_col = f"data_{dataset}_train_settings"
# df_plot = df_out.loc[
#     (df_out[ood_col] == ood_faults)
#     & (df_out[tr_set_col] == train_settings)
# ]
# %%
df_plot = df_out
bw_col = 'model_GNN_graph_bandwidth'
df_plot[bw_col] = df_plot[bw_col].astype(float)
temp_col = 'loss_embedding_temperature'
df_plot[temp_col] = df_plot[temp_col].astype(float)

# %%
df_plot = df_plot.loc[df_plot['seed'] != '71',]
df_plot = df_plot.loc[df_plot[temp_col] != 0.02,]
df_plot = df_plot.loc[df_plot[bw_col] != 0.2,]

# %%
# Test accuarcy without OOD faults
bw_mean = df_plot.loc[
        (df_plot[temp_col] == 0.1) & (~df_plot['ood']),
        [bw_col, 'clf_test_setting_acc']
    ].groupby(bw_col).mean().reset_index()
bw = bw_mean[bw_col].values
bw_mean_vals = bw_mean['clf_test_setting_acc'].values
bw_std = df_plot.loc[
    (df_plot[temp_col] == 0.1) & (~df_plot['ood']),
    [bw_col, 'clf_test_setting_acc']
    ].groupby(bw_col).std().reset_index()['clf_test_setting_acc'].values
temp_mean = df_plot.loc[
    (df_plot[bw_col] == 2) & (~df_plot['ood']),
    [temp_col, 'clf_test_setting_acc']
].groupby(temp_col).mean().reset_index()
temp = temp_mean[temp_col].values
temp_mean_vals = temp_mean['clf_test_setting_acc'].values
temp_std = df_plot.loc[
    (df_plot[bw_col] == 2) & (~df_plot['ood']),
    [temp_col, 'clf_test_setting_acc']
].groupby(temp_col).std().reset_index()['clf_test_setting_acc'].values

# %%
# OOD detection
bw_mean_ood = df_plot.loc[
        (df_plot[temp_col] == 0.1) & (df_plot['ood']),
        [bw_col, 'ood_train_setting_acc']
    ].groupby(bw_col).mean().reset_index()
bw_ood = bw_mean_ood[bw_col].values
bw_mean_ood_vals = bw_mean_ood['ood_train_setting_acc'].values
bw_std_ood = df_plot.loc[
    (df_plot[temp_col] == 0.1) & (df_plot['ood']),
    [bw_col, 'ood_train_setting_acc']
    ].groupby(bw_col).std().reset_index()['ood_train_setting_acc'].values
temp_mean_ood = df_plot.loc[
    (df_plot[bw_col] == 2) & (df_plot['ood']),
    [temp_col, 'ood_train_setting_acc']
].groupby(temp_col).mean().reset_index()
temp_ood = temp_mean_ood[temp_col].values
temp_mean_ood_vals = temp_mean_ood['ood_train_setting_acc'].values
temp_std_ood = df_plot.loc[
    (df_plot[bw_col] == 2) & (df_plot['ood']),
    [temp_col, 'ood_train_setting_acc']
].groupby(temp_col).std().reset_index()['ood_train_setting_acc'].values

# %%
# Gaussian bandwidth plot
fig, (ax1, ax2) = plt.subplots(figsize=(fig_dim[0], fig_dim[1]*0.8), ncols=2)

ax1.plot(
    bw, bw_mean_vals, marker='o',
    color=colors[0], label=r'acc$_T$'
)
ax1.fill_between(bw, bw_mean_vals - bw_std, bw_mean_vals + bw_std, alpha=0.2)
ax1.legend()
ax1.set_ylim([0.7, 1.0])
ax1.set_xticks(bw)
ax1.grid()
ax1.set_title('Domain Generalization')

ax2.plot(
    bw_ood, bw_mean_ood_vals, marker='o',
    color=colors[1], label=r'acc$_{OOD}$'
)
ax2.fill_between(
    bw_ood, bw_mean_ood_vals - bw_std_ood, bw_mean_ood_vals + bw_std_ood,
    alpha=0.2
)
ax2.legend()
ax2.set_ylim([0.7, 1.0])
ax2.set_xticks(bw_ood)
ax2.grid()
ax2.set_title('Novel Fault Detection')
ax2.set_yticklabels([])

fig.supxlabel(r'Gaussian Kernel Bandwidth $\zeta$', y=-0.05)
fig.supylabel('Accuracy')
fig.savefig("fig_sensitivity_analysis_bandwidth.pdf", bbox_inches='tight')

# Temperature plot
fig, (ax1, ax2) = plt.subplots(figsize=(fig_dim[0], fig_dim[1]*0.8), ncols=2)

ax1.plot(
    temp, temp_mean_vals, marker='o',
    color=colors[0], label=r'acc$_T$'
)
ax1.fill_between(
    temp, temp_mean_vals - temp_std, temp_mean_vals + temp_std, alpha=0.2
)
ax1.legend()
ax1.set_ylim([0.7, 1.0])
ax1.set_xticks(temp)
ax1.grid()
ax1.set_title('Domain Generalization')

ax2.plot(
    temp_ood, temp_mean_ood_vals, marker='o',
    color=colors[1], label=r'acc$_{OOD}$'
)
ax2.fill_between(
    temp_ood,
    temp_mean_ood_vals - temp_std_ood,
    temp_mean_ood_vals + temp_std_ood,
    alpha=0.2
)
ax2.legend()
ax2.set_ylim([0.7, 1.0])
ax2.set_xticks(temp_ood)
ax2.grid()
ax2.set_title('Novel Fault Detection')
ax2.set_yticklabels([])

fig.supxlabel(r'Temperature $\tau$', y=-0.05)
fig.supylabel('Accuracy')
fig.savefig("fig_sensitivity_analysis_temperature.pdf", bbox_inches='tight')
