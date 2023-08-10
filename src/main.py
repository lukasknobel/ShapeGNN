import datetime
import json
import pathlib
import random
import sys
import os
import time

cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(cur_dir))

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.TrainEvaluationHandling import train_model, test_model
from src.util import find_existing_model, set_seed, match_str_and_enum, prepare_dict_for_summary, get_param_count, \
    parse_args, construct_param_dicts
from src.data_handling.DatasetHandler import SupportedDatasets, Splits
from src.models.ModelFactory import Models, create_model
from src.models.GoGModel import GoGModel
from src.Scheduler import CosineLRScheduler
from data_handling.ImgGraphDataset import ImgGraphDataset


def main(args, model_type=Models.shape_gnn, dataset_type: SupportedDatasets = SupportedDatasets.CIFAR10):

    start = time.time()

    if args.disable_val:
        print('disable_val is True, setting return_last=True and patience=0')
        args.return_last = True
        args.patience = 0

    # check arguments
    if (args.conv_layer_type == 'transformerconv' or args.conv_layer_type == 'gatv2conv') and args.use_edge_weights:
        raise ValueError("edge weights not supported for transformerconv and gatv2conv due to always utilising sparsity due to memory constraints")
    if args.conv_layer_type =='egnn' and args.num_pool_blocks > 0:
        raise ValueError("EGNN is not supported with graph pooling due to it creating its own edges")
    if args.use_edge_weights and args.num_pool_blocks > 0:
        raise ValueError("Edge weights are not supported with graph pooling")

    # get hyperparameter dictionaries
    hparams, graph_params, prep_params = construct_param_dicts(args, model_type, dataset_type)
    print(f'Configuration: {json.dumps(hparams, indent=4)}')

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)

    # define the paths to the base, results and data directory
    base_dir = pathlib.Path(__file__).parent.parent
    result_dir = os.path.join(base_dir, 'runs')
    if args.data_dir == '':
        data_dir = os.path.join(base_dir, 'data')
    else:
        data_dir = args.data_dir
    # create results dictionary if necessary
    if not os.path.isdir(result_dir):
        print('Creating results dir')
        os.mkdir(result_dir)
    print("data directory: ", data_dir)
    print("original data directory: ", args.orig_data_dir)

    # create datasets for different setups
    raw_train_dataset = ImgGraphDataset(base_root_dir=data_dir, data_dir=args.orig_data_dir,
                                        dataset_type=dataset_type, split=Splits.train, prep_params=prep_params,
                                        graph_params=graph_params, only_create_dataset=args.only_create_dataset, max_out_ram_for_preprocessing=args.max_out_ram_for_preprocessing,
                                        statistics_file_type=None)
    test_dataset = ImgGraphDataset(base_root_dir=data_dir, data_dir=args.orig_data_dir, dataset_type=dataset_type,
                                   split=Splits.test, prep_params=prep_params, graph_params=graph_params,
                                   only_create_dataset=args.only_create_dataset, max_out_ram_for_preprocessing=args.max_out_ram_for_preprocessing, statistics_file_type='train')
    if args.only_create_dataset:
        return

    # determine sample indices for training and validation
    if args.disable_val:
        train_idxs = torch.arange(len(raw_train_dataset), dtype=torch.int64)
        val_idxs = None
    else:
        train_val_percentage = 0.9
        train_size = int(len(raw_train_dataset) * train_val_percentage)
        train_idxs = np.random.choice(np.arange(len(raw_train_dataset), dtype=np.int64), size=train_size, replace=False)
        val_mask = np.ones(len(raw_train_dataset), dtype=bool)
        val_mask[train_idxs] = 0
        val_idxs = np.arange(len(raw_train_dataset), dtype=np.int64)[val_mask]
        train_idxs = torch.tensor(train_idxs)
        val_idxs = torch.tensor(val_idxs)
    num_classes = len(raw_train_dataset.classes)

    print(f'TIME, dataset creation: {time.time()-start:.2f}')

    # determine number of input features
    in_features = 4
    num_shape_node_features = 2
    if hparams['graph_of_graphs']:
        in_features += args.latent_dim
    if hparams['use_stddev']:
        in_features += 3
    if hparams['no_size']:
        in_features -= 1
    if hparams.get('superpixel_rotation_information', False):
        in_features += 1

    # create model
    model = create_model(model_type, num_classes, in_features,
                         linear_dropout=args.linear_dropout, batch_norm_momentum=args.batch_norm_momentum,
                         act_fn=args.act_fn, conv_layer_type=args.conv_layer_type, agg=args.agg, pool=args.pool,
                         num_linear_layers_mult=args.num_linear_layers_mult,
                         num_res_blocks=args.num_res_blocks, residual_type=args.residual_type,
                         num_hidden=args.hidden_dim, num_pool_blocks=args.num_pool_blocks, pool_factor=args.pool_factor,
                         pool_hidden_multiplier=args.pool_hidden_multiplier, graph_pool_type=args.graph_pool_type,
                         ign_position=args.ign_position)
    if hparams['graph_of_graphs']:
        model = GoGModel(num_shape_node_features, latent_dim=args.latent_dim,
                                          gnn_model=model, enc_hidden_dim=args.enc_hidden_dim,
                                          gog_batch_norm=args.gog_batch_norm,
                                          enc_conv_layer_type=args.enc_conv_layer_type,
                                          enc_linear_layers_mult=args.enc_linear_layers_mult,
                                          enc_num_res_blocks=args.enc_num_res_blocks,
                                          enc_residual_type=args.enc_residual_type)
    print('#params: ', get_param_count(model))

    # create optimizers and scheduler
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)
        scheduler = None
        hparams['optimizer'] = 'Adam'
        hparams['scheduler'] = 'None'
    elif args.optimizer == 'AdamWeight':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr, weight_decay=5e-6)
        scheduler = None
        hparams['optimizer'] = 'AdamWeight'
        hparams['scheduler'] = 'None'
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.model_lr)
        scheduler = None
        hparams['optimizer'] = 'AdamW'
        hparams['scheduler'] = 'None'
    elif args.optimizer == 'AdamWWeight':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.model_lr, weight_decay=0.9e-2)
        scheduler = None
        hparams['optimizer'] = 'AdamWWeight'
        hparams['scheduler'] = 'None'
    elif args.optimizer == 'AdamAnnealing':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10)
        hparams['optimizer'] = 'Adam'
        hparams['scheduler'] = 'CosineAnnealing'
    elif args.optimizer == 'AdamWCosine':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.model_lr, weight_decay=0.05)
        t_initial = args.t_initial
        if t_initial == 0:
            t_initial = args.epochs
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=1e-5,
            warmup_lr_init=args.model_lr,
            warmup_t=args.lr_warmup_t,
            k_decay=getattr(args, 'lr_k_decay', 1.0),
        )
        hparams['optimizer'] = 'AdamW'
        hparams['scheduler'] = 'CosineScheduler'
        hparams['t_initial'] = t_initial
        hparams['warmup'] = args.lr_warmup_t
    else:
        raise ValueError(f'Unknown optimizer "{args.optimizer}"')

    model = model.to(device)

    # determine (existing or new) model directory
    stored_model_dir = find_existing_model(result_dir, hparams)
    model_name = model_type.name
    if stored_model_dir is None:
        # create a new model directory that is not already used
        if args.model_folder_postfix != '':
            model_name += '_' + args.model_folder_postfix
        model_name += '_'+datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        model_dir = os.path.join(result_dir, model_name)
        i = 1
        while os.path.isdir(model_dir):
            model_dir = os.path.join(result_dir, model_name + f'_{i}_{random.randint(0, 1e6)}')
            i += 1
        print('Creating model dir')
        os.mkdir(model_dir)
    else:
        model_dir = stored_model_dir
    writer_dir = os.path.join(model_dir, 'tb_logs')
    hparams_file = os.path.join(model_dir, 'hparams.json')

    # save hyperparameters
    with open(hparams_file, 'w') as f:
        json.dump(hparams, f, indent=4)

    print(f'TIME, model creation: {time.time() - start:.2f}')

    summary_writer = SummaryWriter(writer_dir)

    # load or train model
    checkpoint_name = os.path.join(model_dir, 'model.model')
    if os.path.isfile(checkpoint_name):
        print('Loading trained model')
        state_dict = torch.load(checkpoint_name, device)
        model.load_state_dict(state_dict)
    else:
        print('Training model')
        model = train_model(model, optimizer, scheduler, args.batch_size, args.epochs, args.patience,
                            raw_train_dataset, train_idxs, val_idxs, checkpoint_name, summary_writer, device,
                            graph_of_graphs=hparams['graph_of_graphs'], label_smoothing=args.label_smoothing,
                            clip_norm=args.clip_norm, return_last=args.return_last, num_workers=args.num_workers,
                            disable_tqdm=args.disable_tqdm)

    print(f'TIME, model training/loading: {time.time() - start:.2f}')

    # test model
    test_result_file_name = os.path.join(model_dir, 'test_results.json')
    if os.path.isfile(test_result_file_name):
        print('Loading test results')
        with open(test_result_file_name, 'r') as f:
            test_results = json.load(f)
    else:
        print('Testing model')
        test_results = test_model(model, args.batch_size, test_dataset, hparams['graph_of_graphs'],
                                  args.label_smoothing, device, args.seed, num_workers=args.num_workers)
        with open(test_result_file_name, 'w') as f:
            json.dump(test_results, f, indent=6)
        summary_writer.add_hparams(prepare_dict_for_summary(hparams), {'hparam/accuracy': test_results['accuracy-1'], 'hparam/accuracy-5': test_results['accuracy-5']})
    print(f'TIME, model testing: {time.time() - start:.2f}')
    print(f'Test results: {test_results["loss"]:.3f}(loss), {test_results["accuracy-1"]:.3f}(accuracy-1), {test_results["accuracy-5"]:.3f}(accuracy-5), {test_results["time"]:.2f}s')

    summary_writer.flush()
    summary_writer.close()
    print(f'TIME, total: {time.time() - start:.2f}')


if __name__ == '__main__':

    args = parse_args()

    model_type = match_str_and_enum(args.model_name, Models)
    dataset = match_str_and_enum(args.dataset, SupportedDatasets)
    main(args, model_type, dataset)
