import logging
from datetime import datetime
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from data.base import create_emb_dataloader
from data.init_data_loader import init_data_loader
from emb_loss import EmbLoss
from evaluation import evaluate_model_performance
from models.cnn import CnnBase
from models.gnn import GnnBase
from models.cnn_da import CnnDA
from utils import ensure_reproduciblity, yaml_dump


def main(config: dict, save_model: bool = False) -> None:
    '''
    Main function to run the experiment.
    It loads the data, initializes the model, trains it, and evaluates the
    performance.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing the parameters for the experiment.
    save_model : bool, optional
        Whether to save the trained model.
    '''

    # TODO final code
    # anonymize dataset paths

    if config['verbose']:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    else:
        logging.basicConfig(level=logging.WARNING)

    ensure_reproduciblity(config['seed'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = torch.device(f"cuda:{config['cuda_device']}")
        else:
            device = torch.device('cuda')
    # device = torch.device('cpu')
    logging.info(f"Device: {device}")

    # Load data
    data_set_name = config['dataset']
    data_config = config['data'][data_set_name]
    data = init_data_loader(
        data_set_name,
        data_config['path_to_data'],
        config['data']['force_reload'],
        data_config['kg_path'],
        config['seed']
    )
    data.get_euclidean_data(**data_config['load'])

    # Check what parts the model has and init losses
    criterion = {}
    criterion['clf'] = torch.nn.CrossEntropyLoss()
    if (
        ('clf' in config['loss'].keys())
        and ('embedding' in config['loss'].keys())
    ):
        parts = ['embedding', 'clf']
        criterion['embedding'] = EmbLoss(config['loss']['embedding']['name'])
        # Create y embedding based on knowledge graph
        # Needed for all embedding losses for projection dimension
        data.get_kg_emb_target(Path(data_config['kg_path']))
        projection_dim = data.y_kg_emb.shape[1]
    else:
        parts = ['clf']
        # No embedding needed
        projection_dim = None

    if data_set_name == 'CWRU':
        train_settings = list(map(
            lambda rpm: data.rpm_mapping[rpm],
            data_config['train_settings']
        ))
    else:
        train_settings = data_config['train_settings']

    # Create dataloaders and init models
    model_type = config['model']['type']
    model_config = config['model'][model_type]
    if (model_type == 'CNN') or (model_type == 'CNN_DA'):
        # Create euclidean dataloaders and init CNN
        dataloader_train, dataloader_test = data.create_euclidean_dataloaders(
            test_size=1-config['data']['train_size'],
            batch_size=config['data']['batch_size'],
            train_settings=train_settings,
            ood_fault_list=data_config['load']['ood_faults']
        )
        exemp_input = next(iter(dataloader_train))['X'].to(device)
        if model_type == 'CNN':
            model = CnnBase(
                n_classes=data.n_train_classes,
                input_dim=data.X.shape[2],
                input_channels=data.X.shape[1],
                embedding_dim=config['embedding_dim'],
                projection_dim=projection_dim,
                parts=parts,
                dataset=config['dataset'],
                split_type=model_config['split_type']
            ).to(device)
        elif model_type == 'CNN_DA':
            model = CnnDA(
                n_classes=data.n_train_classes,
                n_domains=len(train_settings),
                dataset=config['dataset']
            ).to(device)
    elif model_type == 'GNN':
        # Create graph data loaders and init GNN
        data.transform_to_graph_data(**model_config['graph'])
        dataloader_train, \
            dataloader_test = data.create_graph_dataloaders(
                test_size=1-config['data']['train_size'],
                batch_size=config['data']['batch_size'],
                train_settings=train_settings,
                ood_fault_list=data_config['load']['ood_faults']
            )
        exemp_input = next(iter(dataloader_train)).to(device)
        model = GnnBase(
            n_classes=data.n_train_classes,
            embedding_dim=config['embedding_dim'],
            projection_dim=projection_dim,
            n_sensors=data.n_sensors,
            n_phases=data.n_phases,
            parts=parts,
            dataset=config['dataset'],
            cnn_pre=model_config['cnn_pre'],
        ).to(device)

    del_k = [k for k, v in dataloader_test.items() if len(v) == 0]
    for k in del_k:
        # Datatloader is empty because all settings are used
        # for training and there are no ood faults
        dataloader_test.pop(k, None)

    # Save dataloader for later evaluation
    dataloader_train_eval = dataloader_train
    dataloader_test_eval = dataloader_test

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Set and create logging dir
    loss_str = "clfCE"
    try:
        embloss = config['loss']['embedding']['name']
        loss_str += f"_emb{embloss}"
    except KeyError:
        pass
    if config['dataset'] == 'PHMAP23Spacecraftpropulsion':
        loss_str += f"_split{data_config['load']['split']}"
    log_parent_dir = Path(
        f"experiments/{config['dataset']}"
        f"/{config['experiment_name']}"
    )
    res_dir = (
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        f"_ts{config['data']['train_size']}"
        f"_set{'_'.join([str(s) for s in train_settings])}"
        f"_{config['model']['type']}"
        f"_{loss_str}"
        f"_{config['data']['batch_size']}"
    )
    res_dir = log_parent_dir / Path(res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    # Train separate parts of network
    for i_p, part in enumerate(parts):

        if i_p == 0:
            if config['verbose']:
                logging.info("Model summary:")
                model.get_summary(exemp_input)

        if ('embedding' in parts) and (part == 'clf'):
            # Get embedding and put in dataloader
            _, train_emb_x, \
                _, train_y = model.get_proj_emb_clf_y_from_dataloader(
                    dataloader_train, device, projection=False
                )
            dataloader_train = create_emb_dataloader(
                train_emb_x, train_y, config['data']['batch_size'],
                shuffle=True
            )
            del train_emb_x, train_y
            dataloader_tmp = {}
            for k, v in dataloader_test.items():
                if len(v) > 0:
                    _, test_emb_x, \
                        _, test_y = model.get_proj_emb_clf_y_from_dataloader(
                            v, device, projection=False
                        )
                    dataloader_tmp[k] = create_emb_dataloader(
                        test_emb_x, test_y, config['data']['batch_size'],
                        shuffle=False
                    )
            dataloader_test = dataloader_tmp
            del dataloader_tmp, test_emb_x, test_y

        logging.info(f"Training part: {part}")
        if config['verbose']:
            epoch_progress = range(1, config['loss'][part]['epochs']+1)
        else:
            epoch_progress = tqdm(
                range(1, config['loss'][part]['epochs']+1),
                desc=f"Part: {part}",
                leave=False
            )

        for epoch in epoch_progress:

            model.train_epoch_step(
                data_loader=dataloader_train,
                optimizer=optimizer,
                criterion=criterion[part],
                mode=part
            )

            if epoch % config['eval_interval'] == 0:
                # Evaluate model
                train_loss, train_acc = model.eval_epoch_step(
                    data_loader=dataloader_train,
                    criterion=criterion[part],
                    mode=part
                )
                test_loss, test_acc = {}, {}
                for k, v in dataloader_test.items():
                    if (len(v) > 0) and (not ('ood' in k)):
                        test_loss[k], test_acc[k] = model.eval_epoch_step(
                            data_loader=v,
                            criterion=criterion[part],
                            mode=part
                        )

                train_res = (
                    f"Train - Loss: {train_loss:.2f},"
                )
                if train_acc is not None:
                    train_res += (
                        f" Acc: {train_acc:.3f}"
                    )
                test_res = ''
                for k, v in reversed(test_loss.items()):
                    if not ('ood' in k):
                        test_res += (
                            f" | Test {k} - Loss: {v:.2f},"
                        )
                        if test_acc[k] is not None:
                            test_res += (
                                f" Acc: {test_acc[k]:.3f}"
                            )
                print_res = train_res + test_res
                if not config['verbose']:
                    epoch_progress.set_postfix_str(print_res)
                logging.info(
                    f"Epoch {epoch}/{config['loss'][part]['epochs']}"
                    f" | {print_res}"
                )

    logging.info("Evaluating performance")
    dataloaders = {'train': dataloader_train_eval} | dataloader_test_eval
    metric_dict = evaluate_model_performance(
        model, dataloaders, device, parts
    )

    # Save metrics
    yaml_dump(metric_dict, res_dir / 'metrics.yaml')

    # Save config
    config['model_summary'] = str(model)
    yaml_dump(config, res_dir / 'config.yaml')

    if save_model:
        torch.save(model, res_dir / 'model.pt')


if __name__ == '__main__':
    with open('configs/config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    main(config, save_model=True)
