# %%
import pandas as pd
import yaml
from pathlib import Path
import numpy as np

from functools import partial


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


def create_latex_table(
        dataset, experiment_name, model_names, n_tr_settings, ood_type
):

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

    # Define aggregation function
    def agg_func(x, metric_cols):
        agg_dict = {
            f"{k}_mean": round(x[k].mean()*100, 1) for k in metric_cols
        }
        agg_dict |= {
            f"{k}_std": round(x[k].std()*100, 1) for k in metric_cols
        }
        agg_dict = dict(sorted(agg_dict.items()))
        return pd.Series(agg_dict, index=list(agg_dict.keys()))

    rel_config = [
        'model_type',
        'model_GNN_cnn_pre',
        'model_GNN_graph_split_type',
        'model_CNN_split_type',
        'model_CNN_DA_type',
        'model_CNN_DA_da',
        'model_CNN_DA_mmd',
        'loss_embedding_losses_Emb-KG_params_name',
        'n_tr_settings',
        'ood'
    ]

    # Only keep columns that are in the dataset
    rel_config = [col for col in rel_config if col in df_out.columns]

    # Combine old and new columns
    for col in [
        'ood_train_setting_acc_emb', 'ood_train_setting_ood_acc_emb',
        'ood_train_setting_acc_proj', 'ood_train_setting_ood_acc_proj'
    ]:
        if col in df_out.columns:
            col_one = f"{col}_one"
            if col_one not in df_out.columns:
                df_out[col_one] = np.nan
            df_out.loc[df_out[col_one].isna(), col_one] = df_out.loc[
                df_out[col_one].isna(), col
            ]

    # ood type needed for old experiments
    ood_metrics = [
        f"ood_train_setting_acc",  # _{ood_type}",
        f"ood_train_setting_ood_acc"  # _{ood_type}"
    ]

    te_metrics = [
        "clf_train_setting_acc",
        "clf_test_setting_acc",
    ]
    # te_metrics = [
    #     "emb_clf_train_setting_acc",
    #     "emb_clf_test_setting_acc",
    # ]
    rel_metrics = te_metrics + ood_metrics

    agg_func_partial = partial(agg_func, metric_cols=rel_metrics)
    df_grouped = df_out.groupby(
        rel_config, dropna=False
    ).apply(agg_func_partial).reset_index()

    # Add count
    df_grouped['count'] = df_out.groupby(
        rel_config, dropna=False
    ).size().values

    # # Add CE loss param
    # df_grouped.loc[
    #     df_grouped['loss_embedding_losses_Emb-KG_params_name'].isna(),
    #     'loss_embedding_losses_Emb-KG_params_name'
    # ] = 'CE'

    # Add model_name column
    def get_model_name(row):
        loss = 'loss_embedding_losses_Emb-KG_params_name'
        if (
            (row['model_type'] == 'CNN')
            and (row[loss] == 'CE')
            and (row['model_CNN_split_type'] == 'one')
        ):
            return 'CnnCe'
        if (
            (row['model_type'] == 'CNN')
            and (row[loss] == 'CE')
            and (row['model_CNN_split_type'] == 'phase')
        ):
            return 'CnnPsCe'
        if (
            (row['model_type'] == 'CNN')
            and (row[loss] == 'kg_self_dist')
            and (row['model_CNN_split_type'] == 'one')
        ):
            return 'CnnKg'
        if (
            (row['model_type'] == 'CNN')
            and (row[loss] == 'kg_self_dist')
            and (row['model_CNN_split_type'] == 'phase')
        ):
            return 'CnnPsKg'
        if (
            (row['model_type'] == 'CNN')
            and (row[loss] == 'supcon')
            and (row['model_CNN_split_type'] == 'one')
        ):
            return 'CnnSup'
        if (
            (row['model_type'] == 'CNN')
            and (row[loss] == 'supcon')
            and (row['model_CNN_split_type'] == 'phase')
        ):
            return 'CnnPsSup'
        if (
            (row['model_type'] == 'GNN')
            and (row['model_GNN_cnn_pre'] == 'False')
            and (row['model_GNN_graph_split_type'] == 'phase')
            and (row[loss] == 'kg_self_dist')
        ):
            return 'GnnPsKg'
        if (
            (row['model_type'] == 'GNN')
            and (row['model_GNN_cnn_pre'] == 'True')
            and (row['model_GNN_graph_split_type'] == 'one')
            and (row[loss] == 'kg_self_dist')
        ):
            return 'GnnApKg'
        if (
            (row['model_type'] == 'GNN')
            and (row['model_GNN_cnn_pre'] == 'True')
            and (row['model_GNN_graph_split_type'] == 'phase')
            and (row[loss] == 'CE')
        ):
            return 'GnnPsApCe'
        if (
            (row['model_type'] == 'GNN')
            and (row['model_GNN_cnn_pre'] == 'True')
            and (row['model_GNN_graph_split_type'] == 'phase')
            and (row[loss] == 'supcon')
        ):
            return 'GnnPsApSup'
        if (
            (row['model_type'] == 'GNN')
            and (row['model_GNN_cnn_pre'] == 'True')
            and (row['model_GNN_graph_split_type'] == 'phase')
            and (
                row[loss] == 'kg_self_dist'
            )
        ):
            return 'GnnPsApKg'
        if (
            (row['model_type'] == 'CNN_DA')
            and (
                (row['model_CNN_DA_type'] == 'da')
                or (row['model_CNN_DA_da'] == 'True')
                )
        ):
            return 'DA'
        if (
            (row['model_type'] == 'CNN_DA')
            and (
                (row['model_CNN_DA_type'] == 'mmd')
                or (row['model_CNN_DA_mmd'] == 'True')
                )
        ):
            return 'MMD'

    df_grouped['model_name'] = df_grouped.apply(get_model_name, axis=1)
    df_grouped_te = df_grouped[~df_grouped.ood]
    df_grouped_ood = df_grouped[df_grouped.ood]

    def reshape_df(df, value_cols):
        # Reshape df so that n_tr_settings is part of the columns for the
        # metrics
        df = df.pivot_table(
            index='model_name',
            columns='n_tr_settings',
            values=[
                f"{metric}_mean" for metric in value_cols
            ] + [f"{metric}_std" for metric in value_cols]
        ).reset_index()

        # Flatten the multi-level columns
        df.columns = ['_'.join(map(str, col)).strip('_') for col in df.columns]
        return df

    df_grouped_te = reshape_df(df_grouped_te, te_metrics)
    df_grouped_ood = reshape_df(df_grouped_ood, ood_metrics)

    res_dict = {}
    for m in model_names:
        row_te = df_grouped_te[df_grouped_te.model_name == m]
        row_ood = df_grouped_ood[df_grouped_ood.model_name == m]
        # if "Kg" in m:
        #     te_metrics_rel = te_metrics[-2:]
        # else:
        #     te_metrics_rel = te_metrics[:2]
        str_out = [m]
        for n in n_tr_settings:
            for te_m in te_metrics:
                mean = f"{te_m}_mean_{n}"
                std = f"{te_m}_std_{n}"
                try:
                    str_out.append(
                        f"{float(row_te[mean].iloc[0])}"
                        f"\\tiny{{({float(row_te[std].iloc[0])})}}"
                    )
                except (KeyError, IndexError):
                    str_out.append('-')
            for ood_m in ood_metrics:
                mean = f"{ood_m}_mean_{n}"
                std = f"{ood_m}_std_{n}"
                try:
                    str_out.append(
                        f"{float(row_ood[mean].iloc[0])}"
                        f"\\tiny{{({float(row_ood[std].iloc[0])})}}"
                    )
                except (KeyError, IndexError):
                    str_out.append('-')
        res_dict[m] = ' & '.join(str_out) + ' \\\\'

    tab_str = "\\begin{tabular}{@{}lllllllllllll@{}} \\toprule Model"
    for n in n_tr_settings:
        tab_str += f" & \\multicolumn{{4}}{{c}}{{$|S|={n}$}}"
    tab_str += " \\\\"
    for i, _ in enumerate(n_tr_settings):
        tab_str += f" \\cmidrule(lr){{{2+i*4}-{5+i*4}}}"
    for n in n_tr_settings:
        tab_str += " & $\\accS$ & $\\accT$ & $\\accID$ & $\\accOOD$"
    tab_str += " \\\\ \\midrule "
    for m in model_names:
        tab_str += res_dict[m]
    tab_str += " \\bottomrule \\end{tabular}"

    return tab_str


if __name__ == "__main__":
    # ds = "PHMAP23Spacecraftpropulsion"
    # exp_name = "2025-10-29_domain_adversarial_training"
    # exp_name = "2025-10-31_mmd_training_add_layer"
    # ds = "CWRU"
    # exp_name = "2025-10-30_da_training_add_layer"
    # exp_name = "2025-10-31_mmd_training_add_layer"
    ds = "PHM22Rockdrill"
    exp_name = "2025-10-29_domain_adversarial"
    # exp_name = "2025-10-31_mmd_training"
    # exp_name = "2025-03-04_christmas_extension_DQ_ST_V_with_christmas_extension_with_supcon"
    if ds == 'PHM22Rockdrill':
        settings = [4, 5, 6]
    else:
        settings = [2, 3, 4]
    models = [
        # "MMD",
        "DA",
        # "CnnCe",
        # "CnnSup",
        # "CnnKg",
        # "CnnPsCe",
        # "CnnPsSup",
        # "CnnPsKg",
        # "GnnPsKg",
        # "GnnApKg",
        # "GnnPsApCe",
        # "GnnPsApSup",
        # "GnnPsApKg",
    ]

    table = create_latex_table(
        ds,
        exp_name,
        models,
        settings,
        ood_type='emb_one'
    )
    print(table)

# %%
