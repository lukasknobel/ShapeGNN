import argparse
import json
import os

import numpy as np
import torch


def prepare_dict_for_summary(hparams_dict):
    new_dict = {}
    for key, value in hparams_dict.items():
        if isinstance(value, dict):
            sub_dict = prepare_dict_for_summary(value)
            for sub_key, sub_value in sub_dict.items():
                new_dict[key+'_'+sub_key] = sub_value
        else:
            new_dict[key] = value
    return new_dict


def match_str_and_enum(str_, enum_, match_case=False):
    for enum_element in enum_:
        if match_case and enum_element.name == str_:
            return enum_element
        elif not match_case and enum_element.name.lower() == str_.lower():
            return enum_element
    raise ValueError(f'Unknown name \"{str_}\" for {enum_.__class__.__name__}')


def find_existing_model(models_dir, hparams, hparams_file='hparams.json'):
    for el in os.listdir(models_dir):
        hparams_file_path = os.path.join(models_dir, el, hparams_file)
        if os.path.isfile(hparams_file_path):
            with open(hparams_file_path) as f:
                stored_hparams = json.load(f)
            if stored_hparams == hparams:
                model_path = os.path.join(models_dir, el)
                print(f'Found saved model: {model_path}')
                return model_path
    return None


def get_number_trainable_parameters(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def accuracy(preds, labels, top_k=(1,5)):
    with torch.no_grad():
        _, top_k_classes = preds.topk(max(top_k), largest=True, sorted=True)
        correct_class = top_k_classes.eq(labels.view(-1, 1))
        accs = []
        for k in top_k:
            acc = correct_class[:, :k].any(dim=-1).float().mean()
            accs.append(acc)
        return accs


def print_epoch_times(results_folder):
    folder_elements = [os.path.join(results_folder, el) for el in os.listdir(results_folder)]
    for el in folder_elements:
        hparams_file_name = os.path.join(el, 'hparams.json')
        train_res_file_name = os.path.join(el, 'model_results.csv')
        if os.path.isfile(hparams_file_name) and os.path.isfile(train_res_file_name):
            data = np.loadtxt(train_res_file_name)
            if data.shape[0] == 4:
                epoch_times = data[-1]
                with open(hparams_file_name, 'r') as f:
                    hparams = json.load(f)
                print_keys = ['model_type', 'use_slic']
                for key, value in hparams.items():
                    if key in print_keys:
                        print(f'{key}: {value}, ', end='')
                print()
                print(f'Median: {np.median(epoch_times)}')


def get_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args():
    """
    Parse command line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='shape_gnn', type=str, help='Name of the model type')
    parser.add_argument('--model_folder_postfix', default='', type=str, help='Postfix added to model folders')
    parser.add_argument('--return_last', action='store_true',
                        help='Use the model of the last epoch instead of the best model on the validation set')
    parser.add_argument('--disable_val', action='store_true',
                        help='Use the whole training set for training and don\'t split it into training and validation')
    parser.add_argument('--linear_dropout', default=0.1, type=float, help='Dropout rate for linear blocks')
    parser.add_argument('--batch_norm_momentum', default=-1, type=float,
                        help='Batch norm momentum used in linear blocks, no batch norm is used if < 0')
    parser.add_argument('--gog_batch_norm', action='store_true',
                        help='Adds a batch norm between the local and global GNN')
    parser.add_argument('--use_edge_weights', action='store_true',
                        help='Use distances between superpixel centroids as edge weights for the global graph')
    parser.add_argument('--use_sub_edge_weights', action='store_true',
                        help='Use distance between nodes in the local graph as edge weights')
    parser.add_argument('--act_fn', default='leakyrelu', type=str,
                        help='Name of the activation function used in the model')
    parser.add_argument('--agg', default='mean', type=str, help='Aggregation used by the graph convolution in the model')
    parser.add_argument('--pool', default='att', type=str, help='Pooling used in the model')
    parser.add_argument('--conv_layer_type', default='edgeconv', type=str,
                        help='Graph convolution type used in the global GNN')
    parser.add_argument('--enc_conv_layer_type', default='edgeconv', type=str,
                        help='Graph convolution type used in the local GNN')
    parser.add_argument('--num_linear_layers_mult', default=1, type=int,
                        help='Multiplier to increase the depth of the linear blocks')
    parser.add_argument('--num_res_blocks', default=1, type=int,
                        help='Multiplier to increase the number of residual blocks')
    parser.add_argument('--num_pool_blocks', default=0, type=int,
                        help='Multiplier to increase the number of graph pooling blocks')
    parser.add_argument('--pool_factor', default=3, type=int,
                        help='Number of nodes that are pooled together into one per graph pooling block')
    parser.add_argument('--pool_hidden_multiplier', default=2, type=float,
                        help='Factor with which to increase the hidden dimension after the graph pool in a graph pooling block')
    parser.add_argument('--graph_pool_type', default='mean', type=str, help='Pooling type used for the graph pooling')
    parser.add_argument('--residual_type', default='add', type=str,
                        help='Type of residual connection used by the model')
    parser.add_argument('--hidden_dim', default=300, type=int, help='Hidden dimensionality of the global GNN')
    parser.add_argument('--enc_linear_layers_mult', default=2, type=int,
                        help='Multiplier to increase the depth of the linear blocks of the local GNN')
    parser.add_argument('--enc_num_res_blocks', default=2, type=int,
                        help='Multiplier to increase the number of residual blocks in the local GNN')
    parser.add_argument('--enc_residual_type', default='add', type=str,
                        help='Type of residual connection used by the local GNN')
    parser.add_argument('--enc_hidden_dim', default=64, type=int,
                        help='Hidden dimensionality of the local GNN, equals latent_dim if 0')
    parser.add_argument('--ign_position', action='store_true', help='Use the ShapeGNN without position encoder')

    parser.add_argument('--dataset', default='cifar10', type=str, help='Name of the dataset')
    parser.add_argument('--data_dir', default='', type=str,
                        help='Path where the processed data is stored. If orig_data_dir is not provided, this is also the path to the original data')
    parser.add_argument('--orig_data_dir', default='', type=str,
                        help='Path to the original data. If not provided, data_dir is used.')
    parser.add_argument('--disable_tqdm', action='store_true', help='Disable TQDM')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers')
    parser.add_argument('--seed', default=832, type=int, help='Seed for pseudorandom numbers')

    parser.add_argument('--use_slic', action='store_true', help='Use SLIC instead of Felzenszwalb')
    parser.add_argument('--graph_of_graphs', action='store_true',
                        help='Use graph-of-graphs data jointly training local and global GNN')
    parser.add_argument('--scale', default=10, type=float, help='Scale parameter for Felzenszwalb determining the threshold for merging superpixels')
    parser.add_argument('--sigma', default=0.8, type=float, help='Parameter for Gaussian smoothing')
    parser.add_argument('--min_size', default=5, type=float,
                        help='Parameter for Felzenszwalb determining the minimum size of a superpixel. Should be an integer unless scale_params_to_img is set, in which case it can also be a float, with the final value after scaling becoming an int')
    parser.add_argument('--compactness', default=30, type=float, help='Parameter for SLIC')
    parser.add_argument('--region_size', default=4, type=int,
                        help='Parameter for SLIC determining the desired width and height of a superpixel')
    parser.add_argument('--approx_epsilon', default=0.02, type=float, help='Parameter for polygon approximation')

    parser.add_argument('--use_stddev', action='store_true', help='Use standard deviation of the colour as a feature')
    parser.add_argument('--no_size', action='store_true', help='Do not use patch size as a node feature')
    parser.add_argument('--norm_centroids', action='store_true', help='Normalise centroid coordinates')
    parser.add_argument('--polar_centroids', action='store_true', help='Use polar coordinates for centroids')
    parser.add_argument('--normalise_shapes', action='store_true', help='Normalise shapes')
    parser.add_argument('--normalise_node_data', action='store_true', help='Normalise node data')
    parser.add_argument('--polar_shapes', action='store_true', help='Use polar coordinates for shapes')
    parser.add_argument('--include_shape_centers', action='store_true', help='Add shape centers to the local graph')
    parser.add_argument('--superpixel_rotation_information', action='store_true',
                        help='Add superpixel rotation information')
    parser.add_argument('--scale_params_to_img', action='store_true',
                        help='Scale Felzenszwalb parameters with image size')
    parser.add_argument('--scale_only_min_size', action='store_true',
                        help='When scale_params_to_img is used, scale only min_size')

    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    parser.add_argument('--epochs', default=300, type=int, help='Maximum number of epochs')
    parser.add_argument('--patience', default=0, type=int, help='Patience for early stopping (ignored if 0)')
    parser.add_argument('--label_smoothing', default=0.1, type=float, help='Label smoothing')
    parser.add_argument('--clip_norm', default=0.0, type=float, help='Clip the gradients, ignored if 0')
    parser.add_argument('--optimizer', default='AdamWCosine', type=str, help='Name of the optimizer setup')
    parser.add_argument('--lr_warmup_t', default=20, type=int, help='Number of epochs for learning rate warm up')
    parser.add_argument('--t_initial', default=0, type=int,
                        help='Number of epochs for learning rate resets, ignored if 0')
    parser.add_argument('--model_lr', default=0.001, type=float, help='Learning rate for the model')
    parser.add_argument('--latent_dim', default=5, type=int, help='Latent dimensionality for shape encoding')

    parser.add_argument('--only_create_dataset', action='store_true', help='Create the dataset representation and exit')
    parser.add_argument('--max_out_ram_for_preprocessing', action='store_true',
                        help='Whether to allocate the maximum amount of RAM available minus 5GB for the dataset file. This prevents the C++ allocator from doubling the memory every time more space is needed which might go over the needed allocation if the dataset is too big')

    args = parser.parse_args()
    return args


def construct_param_dicts(args, model_type, dataset_type):
    """
    Construct dictionaries from arguments.
    """

    prep_params = {
        'use_slic': args.use_slic,
        'scale': args.scale,
        'sigma': args.sigma,
        'min_size': args.min_size,
        'compactness': args.compactness,
        'region_size': args.region_size,
        'approx_epsilon': args.approx_epsilon,
        'scale_params_to_img': args.scale_params_to_img,
        'scale_only_min_size': args.scale_only_min_size

    }

    hparams = {
        'model_type': model_type.name, 'dataset': dataset_type.name, 'batch_size': args.batch_size,
        'epochs': args.epochs, 'patience': args.patience, 'seed': args.seed,
        'return_last': args.return_last, 'disable_val': args.disable_val,
        'optimization': {
            'optimization-setup': args.optimizer, 'lr': args.model_lr, 'clip_norm': args.clip_norm,
            'label_smoothing': args.label_smoothing
        },
        'linear_dropout': args.linear_dropout,
        'batch_norm_momentum': args.batch_norm_momentum,
        'act_fn': args.act_fn,
        'conv_layer_type': args.conv_layer_type,
        'num_linear_layers_mult': args.num_linear_layers_mult,
        'num_res_blocks': args.num_res_blocks,
        'residual_type': args.residual_type,
        'hidden_dim': args.hidden_dim,
        'agg': args.agg,
        'pool': args.pool,
        'num_pool_blocks': args.num_pool_blocks,
        'pool_factor': args.pool_factor,
        'pool_hidden_multiplier': args.pool_hidden_multiplier,
        'graph_pool_type': args.graph_pool_type,
        'ign_position': args.ign_position
    }
    graph_params = {
        'use_edge_weights': args.use_edge_weights,
        'use_stddev': args.use_stddev,
        'no_size': args.no_size,
        'norm_centroids': args.norm_centroids,
        'polar_centroids': args.polar_centroids,
        'normalise_node_data': args.normalise_node_data,
        'graph_of_graphs': args.graph_of_graphs
    }

    graph_params['return_pool_block_data'] = True if args.num_pool_blocks is not None else False

    hparams['preprocessing'] = prep_params
    if args.graph_of_graphs:
        hparams['latent_dim'] = args.latent_dim
        hparams['gog_batch_norm'] = args.gog_batch_norm
        hparams['enc_conv_layer_type'] = args.enc_conv_layer_type
        hparams['enc_linear_layers_mult'] = args.enc_linear_layers_mult
        hparams['enc_num_res_blocks'] = args.enc_num_res_blocks
        hparams['enc_residual_type'] = args.enc_residual_type
        hparams['enc_hidden_dim'] = args.enc_hidden_dim

        graph_params['use_sub_edge_weights'] = args.use_sub_edge_weights
        graph_params['superpixel_rotation_information'] = args.superpixel_rotation_information
        graph_params['normalise_shapes'] = args.normalise_shapes
        graph_params['polar_shapes'] = args.polar_shapes
        graph_params['include_shape_centers'] = args.include_shape_centers

    for option, value in graph_params.items():
        hparams[option] = value

    return hparams, graph_params, prep_params