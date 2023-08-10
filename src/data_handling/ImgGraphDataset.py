import os

from torch.utils.data import Dataset
import subprocess

from src.data_handling.DatasetHandler import Splits, SupportedDatasets, get_dataset_dir, is_preprocessed_data_available, make_dataset_available_for_preprocessing
import numpy as np
import torch
from src.data_handling.GraphData import GraphData
from torch import Tensor
from typing import Dict
from torch_sparse import SparseTensor
import psutil

# NOTE: the code below is all vectorised for efficiency and caching of arbitrary transformations is done for efficiency as well
# so variable names and the operations involved might not be very readable, but the complexity is needed solely for efficiency
# and the fact that we are constructing and normalising and batching the input for all image graphs at the same time

# generates a tensor that can index a tensor starting from offsets and going up to length
# used for doing a vectorised select of multiple parts of the image graphs (such as the contour data)
# Example: offsets: [1,6], lengths: [2,4]
# Returned: [1,2,6,7,8,9]
def get_consecutive_idxs(offsets: Tensor, lengths: Tensor):
    lengths_no_last = lengths[:-1]
    cumsum_query_lengths = torch.cumsum(lengths_no_last,0)
    lengths_total = cumsum_query_lengths[-1]+lengths[-1]
    basis = torch.repeat_interleave(offsets-1,lengths, output_size = lengths_total)
    counting = torch.ones(lengths_total,dtype=torch.long)
    counting[cumsum_query_lengths]=-lengths_no_last+1
    final_idxs=basis+torch.cumsum(counting,0)
    return final_idxs

def get_data(given_idxs: Tensor, offsets: Tensor, lengths: Tensor, statistics: Tensor, data: Tensor, graph_params: Dict[str, bool]):

    idxs=torch.sort(given_idxs)[0]
    query_offsets=torch.index_select(offsets,0,idxs)
    query_lengths = torch.index_select(lengths,0,idxs)
    # all the data for the image graphs corresponding to the given indices
    all_data = torch.index_select(data,0,get_consecutive_idxs(query_offsets, query_lengths))
    

    input_idxs_length = idxs.shape[0]
    new_offsets = torch.cat((torch.tensor([0]),torch.cumsum(query_lengths[:-1],0)))
    # getting the data statistics that will inform further fetching and batching
    general_stats = torch.index_select(all_data,0,get_consecutive_idxs(new_offsets, torch.full(idxs.shape,6))).to(torch.long).view(input_idxs_length,6)
    y = general_stats[:,0].contiguous()
    nr_superpixels = general_stats[:,1].contiguous()
    num_edges = general_stats[:,2].contiguous()
    img_height = general_stats[:,3].contiguous()
    img_width = general_stats[:,4].contiguous()
    all_contours_size = general_stats[:,5].contiguous()

    double_num_edges = 2*num_edges

    node_features_length_addition = 7*nr_superpixels
    shape_length_offset = 6+new_offsets
    biggest_distance_shape_idx_offset = shape_length_offset+nr_superpixels
    node_features_offset = biggest_distance_shape_idx_offset+nr_superpixels
    edges_first_offset = node_features_offset+node_features_length_addition
    edges_second_offset = edges_first_offset+double_num_edges

    
    # get the actual data as well
    shape_lengths = torch.index_select(all_data,0,get_consecutive_idxs(shape_length_offset, nr_superpixels)).to(torch.long)
    x = torch.index_select(all_data,0,get_consecutive_idxs(node_features_offset, node_features_length_addition))
    x = x.view(x.shape[0]//7,7)
    used_statistics = statistics.view(statistics.shape[0]//2,2)[[0,1,2,3,4,5,6],:]
    # select superpixel features based on given options
    if not graph_params['use_stddev'] and graph_params['no_size']:
        x = x[:,[1,2,3]]
        used_statistics = used_statistics[[1,2,3],:]
    elif graph_params['use_stddev'] and graph_params['no_size']:
        x = x[:,[1,2,3,4,5,6]]
        used_statistics = used_statistics[[1,2,3,4,5,6],:]
    elif not graph_params['use_stddev'] and not graph_params['no_size']:
        x = x[:,[0,1,2,3]]
        used_statistics = used_statistics[[0,1,2,3],:]

    # create the edges in the way desired by the Pytorch Geometric batching
    actual_cumsum_nr_superpixels = torch.cumsum(nr_superpixels,0)
    nr_superpixels_size = actual_cumsum_nr_superpixels[-1]
    inc_edge_index = torch.repeat_interleave(torch.cat((torch.tensor([0]),actual_cumsum_nr_superpixels[:-1])), double_num_edges)
    edge_index_first = torch.index_select(all_data,0,get_consecutive_idxs(edges_first_offset, double_num_edges)).to(torch.long)+inc_edge_index
    edge_index_second = torch.index_select(all_data,0,get_consecutive_idxs(edges_second_offset, double_num_edges)).to(torch.long)+inc_edge_index
    

    shapes = torch.index_select(all_data,0,get_consecutive_idxs(edges_second_offset+double_num_edges, all_contours_size*2+nr_superpixels*2))
    actual_cumsum_shape_lengths = torch.cumsum(shape_lengths,0)
    cumsum_shape_lengths = torch.cat((torch.tensor([0]),actual_cumsum_shape_lengths[:-1]))
    double_cumsum_shape_lengths = cumsum_shape_lengths*2
    shape_lengths_shape = shape_lengths.shape[0]
    arange_shape_lengths = torch.arange(shape_lengths_shape)

    centroids = torch.index_select(shapes,0,get_consecutive_idxs(double_cumsum_shape_lengths+2*(arange_shape_lengths+shape_lengths), torch.full(double_cumsum_shape_lengths.shape, 2)))
    # create the batch in the way desired by the Pytorch Geometric batching
    batch = torch.repeat_interleave(torch.arange(input_idxs_length),nr_superpixels, output_size=nr_superpixels_size)
    centroids_euclid = centroids.view(shape_lengths_shape,2).contiguous()
    centroid_y = centroids_euclid[:,0]
    centroid_x = centroids_euclid[:,1]
    pos = centroids_euclid
    edge_norm_used_pos = centroids_euclid
    # normalise/convert centroids according to the given options
    if graph_params['norm_centroids'] or graph_params['polar_centroids']:
        centroid_y_diff = centroid_y-torch.repeat_interleave((img_height-1)/2, nr_superpixels, output_size=nr_superpixels_size)
        centroid_x_diff = centroid_x-torch.repeat_interleave((img_width-1)/2, nr_superpixels, output_size=nr_superpixels_size)
        centroid_dist = torch.sqrt(centroid_y_diff**2+(centroid_x_diff)**2)
        centroid_angle = torch.atan2(centroid_y_diff, centroid_x_diff)
        if graph_params['norm_centroids']:
            centroid_dist/=torch.repeat_interleave(torch.sqrt(img_height*img_height+img_width*img_width), nr_superpixels, output_size=nr_superpixels_size)
        edge_norm_used_pos = torch.stack((centroid_dist*torch.sin(centroid_angle), centroid_dist*torch.cos(centroid_angle))).t().contiguous()
        if graph_params['polar_centroids']:
            pos = torch.stack((centroid_dist,(centroid_angle+torch.pi)/(torch.pi*2))).t().contiguous()
        else:
            pos = edge_norm_used_pos

    # compute edge weights
    if graph_params['use_edge_weights']:
        edge_attr = (edge_norm_used_pos[edge_index_first]-edge_norm_used_pos[edge_index_second]).pow(2).sum(1).sqrt()
    else:
        edge_attr = None

    # always use sparse format to save memory and computation due to the sparsity of the input
    sparse_size = edge_index_first[-1]+1
    adj_t = SparseTensor(row=edge_index_first, col=edge_index_second, value=edge_attr, sparse_sizes=(sparse_size, sparse_size), is_sorted=True, trust_data=True)
    shape_lengths_size = actual_cumsum_shape_lengths[-1]

    # for graph of graphs we need to also compute the graphs for the contours
    if graph_params['graph_of_graphs']:
        shape_lengths_plus_one = shape_lengths+1
        shape_lengths_plus_one_size = shape_lengths_size+shape_lengths_shape
        used_shapes = shapes
        edge_norm_used_shapes = shapes
        # normalise node features (which are positions) according to the given options
        if graph_params['normalise_shapes'] or graph_params['polar_shapes'] or graph_params['superpixel_rotation_information']:
            used_shapes = used_shapes.view(shapes.shape[0]//2,2)
            shape_centroids_y = torch.repeat_interleave(centroid_y, shape_lengths_plus_one, output_size=shape_lengths_plus_one_size)
            shape_centroids_x = torch.repeat_interleave(centroid_x, shape_lengths_plus_one, output_size=shape_lengths_plus_one_size)
            used_shapes_y_diff = used_shapes[:,0]-shape_centroids_y
            used_shapes_x_diff = used_shapes[:,1]-shape_centroids_x
            shape_centroid_dist = torch.sqrt(used_shapes_y_diff**2+used_shapes_x_diff**2)
            shape_centroid_angle = torch.atan2(used_shapes_y_diff, used_shapes_x_diff)
            if graph_params['normalise_shapes'] or graph_params['superpixel_rotation_information']:
                cumsum_idx_biggest_distance = torch.index_select(all_data,0,get_consecutive_idxs(biggest_distance_shape_idx_offset, nr_superpixels)).to(torch.long)+cumsum_shape_lengths+arange_shape_lengths
            else:
                cumsum_idx_biggest_distance = None
            if graph_params['superpixel_rotation_information']:
                biggest_angle = shape_centroid_angle[cumsum_idx_biggest_distance]
                shape_centroid_angle-=torch.repeat_interleave(biggest_angle, shape_lengths_plus_one, output_size=shape_lengths_plus_one_size)
                shape_centroid_angle[actual_cumsum_shape_lengths+arange_shape_lengths] = 0
                if graph_params['normalise_node_data']:
                    used_angle = biggest_angle.unsqueeze(1)
                else:
                    used_angle = (biggest_angle.unsqueeze(1)+torch.pi)/(torch.pi*2)
                x = torch.hstack((x,used_angle))
                used_statistics = torch.vstack((used_statistics, statistics.view(statistics.shape[0]//2,2)[7,:]))
            # for normalisation we need to also use the indices of the node with the highest distance from the center in a superpixel
            if graph_params['normalise_shapes']:
                biggest_distance = used_shapes[cumsum_idx_biggest_distance]
                shape_norm_distances = torch.sqrt((torch.repeat_interleave(biggest_distance[:,0], shape_lengths_plus_one, output_size=shape_lengths_plus_one_size)-
                    shape_centroids_y)**2+(torch.repeat_interleave(biggest_distance[:,1], shape_lengths_plus_one, output_size=shape_lengths_plus_one_size)-shape_centroids_x)**2)
                shape_norm_distances[shape_norm_distances==0]=1
                shape_centroid_dist/=shape_norm_distances
            edge_norm_used_shapes = torch.stack((shape_centroid_dist*torch.sin(shape_centroid_angle), shape_centroid_dist*torch.cos(shape_centroid_angle))).t().flatten().contiguous()
            if graph_params['polar_shapes']:
                used_shapes = torch.stack((shape_centroid_dist, (shape_centroid_angle+torch.pi)/(torch.pi*2))).t().contiguous()
            else:
                used_shapes = edge_norm_used_shapes

            used_shapes=used_shapes.flatten()
        
        double_shape_lengths = shape_lengths*2
        # since the (0,0) centers of the shapes are interspersed in the contour data, we need to filter them out if the center option is not used
        if not graph_params['include_shape_centers']:
            edge_norm_idxs = get_consecutive_idxs(double_cumsum_shape_lengths+2*arange_shape_lengths, double_shape_lengths)
            sub_x = torch.index_select(used_shapes,0,edge_norm_idxs).view(shape_lengths_size,2)
        else:
            sub_x = used_shapes.view(edge_norm_used_shapes.shape[0]//2,2)

        # for the shape edges, we generate them dynamically so that we don't have to store them in the dataset
        # and thus the size of the dataset is greatly reduced
        # since they have to be sorted in a way specific to the sparse tensor representation (first by row then by column)
        # we need to do some pattern-based vectorised generation for efficiency
        # these are mostly constructing the beginning, middle, and end parts of each contour representation vectorised
        # then merging them together
        first_begin_sub_edges = cumsum_shape_lengths+1
        second_begin_sub_edges = cumsum_shape_lengths+shape_lengths-1
        first_end_sub_edges = cumsum_shape_lengths
        second_end_sub_edges = second_begin_sub_edges-1

        non_zero_middle = shape_lengths != 2
        shape_lengths_non_zero_middle_minus_two = shape_lengths[non_zero_middle]-2
        
        if graph_params['include_shape_centers']:
            second_begin_sub_edges+=arange_shape_lengths
            third_begin_sub_edges = second_begin_sub_edges+1
            cumsum_arange_shape_lengths = cumsum_shape_lengths+arange_shape_lengths
            cumsum_arange_shape_lengths_non_zero_middle = cumsum_arange_shape_lengths[non_zero_middle]
            first_middle_sub_edges = get_consecutive_idxs(cumsum_arange_shape_lengths_non_zero_middle,shape_lengths_non_zero_middle_minus_two)
            sub_edges_first_cnts = torch.full((shape_lengths_size+shape_lengths_shape,), 3)
            sub_edges_first_cnts[cumsum_arange_shape_lengths+shape_lengths] = shape_lengths
            sub_edges_first = torch.repeat_interleave(torch.arange(shape_lengths_size+shape_lengths_shape), sub_edges_first_cnts, output_size=shape_lengths_size*4)
            sub_edges_second = torch.zeros(shape_lengths_size*4, dtype=torch.long)

            center_edge_offset = cumsum_shape_lengths*4
            center_edge_offset_triple_shape_lengths = center_edge_offset+shape_lengths*3
            
            sub_edges_second[center_edge_offset] = first_begin_sub_edges+arange_shape_lengths
            sub_edges_second[center_edge_offset+1] = second_begin_sub_edges
            sub_edges_second[center_edge_offset+2] = third_begin_sub_edges
            sub_edges_second[get_consecutive_idxs(center_edge_offset[non_zero_middle]+3, shape_lengths_non_zero_middle_minus_two*3)] = torch.stack((first_middle_sub_edges, first_middle_sub_edges+2, 
                torch.repeat_interleave(second_begin_sub_edges[non_zero_middle]+1, shape_lengths_non_zero_middle_minus_two))).t().flatten()
            sub_edges_second[center_edge_offset_triple_shape_lengths-3] = first_end_sub_edges+arange_shape_lengths
            sub_edges_second[center_edge_offset_triple_shape_lengths-2] = second_end_sub_edges+arange_shape_lengths
            sub_edges_second[center_edge_offset_triple_shape_lengths-1] = third_begin_sub_edges
            sub_edges_second[get_consecutive_idxs(center_edge_offset_triple_shape_lengths, shape_lengths)]=get_consecutive_idxs(cumsum_arange_shape_lengths, shape_lengths)
        else:
            cumsum_shape_lengths_non_zero_middle = cumsum_shape_lengths[non_zero_middle]
            sub_edges_first = torch.repeat_interleave(torch.arange(shape_lengths_size), torch.full((shape_lengths_size,), 2), output_size=2*shape_lengths_size)
            sub_edges_second = torch.zeros(shape_lengths_size*2, dtype=torch.long)
            double_cumsum_double_lengths = double_cumsum_shape_lengths+double_shape_lengths
            sub_edges_second[double_cumsum_shape_lengths]=first_begin_sub_edges
            sub_edges_second[double_cumsum_shape_lengths+1] = second_begin_sub_edges
            sub_edges_second[double_cumsum_double_lengths-2] = first_end_sub_edges
            sub_edges_second[double_cumsum_double_lengths-1] = second_end_sub_edges
            sub_edges_second[get_consecutive_idxs(double_cumsum_shape_lengths[non_zero_middle]+2, shape_lengths_non_zero_middle_minus_two*2)] = torch.stack((
                get_consecutive_idxs(cumsum_shape_lengths_non_zero_middle,shape_lengths_non_zero_middle_minus_two), 
                get_consecutive_idxs(cumsum_shape_lengths_non_zero_middle+2,shape_lengths_non_zero_middle_minus_two))).t().flatten()

        # compute edge weights if needed
        if graph_params['use_sub_edge_weights']:
            if not graph_params['include_shape_centers']:
                edge_norm_sub_x = torch.index_select(edge_norm_used_shapes,0,edge_norm_idxs).view(shape_lengths_size,2)
            else:
                edge_norm_sub_x = edge_norm_used_shapes.view(edge_norm_used_shapes.shape[0]//2,2)
            sub_edges_attr = (edge_norm_sub_x[sub_edges_first]-edge_norm_sub_x[sub_edges_second]).pow(2).sum(1).sqrt()
        else:
            sub_edges_attr = None

        sub_sparse_size = sub_edges_first[-1]+1
        # construct the adjacency matric batched sparse representation for the contour input
        sub_adj_t = SparseTensor(row=sub_edges_first, col=sub_edges_second, value=sub_edges_attr, sparse_sizes=(sub_sparse_size, sub_sparse_size), is_sorted=True, trust_data=True)
        if graph_params['include_shape_centers']:
            sub_batch = torch.repeat_interleave(arange_shape_lengths, shape_lengths_plus_one, output_size=shape_lengths_plus_one_size)
        else:
            sub_batch = torch.repeat_interleave(arange_shape_lengths, shape_lengths, output_size=shape_lengths_size)
    else:
        sub_x = None
        sub_adj_t = None
        sub_batch = None

    # normalise superpixel features if required
    if graph_params['normalise_node_data']:
        x = (x-used_statistics[:,0])/used_statistics[:,1]

    if graph_params['return_pool_block_data']:
        edge_index = torch.vstack((edge_index_first, edge_index_second))
        batch_lengths = nr_superpixels
        edge_batch = torch.repeat_interleave(torch.arange(num_edges.shape[0]), double_num_edges)
    else:
        edge_index = None
        batch_lengths = None
        edge_batch = None

    # anything that was not needed according to the parameters given for the input of the model is None, so that we save memory
    return GraphData(x=x, adj_t=adj_t, pos=pos, y=y,
                     sub_x=sub_x, sub_adj_t=sub_adj_t, batch=batch,
                     sub_batch=sub_batch, edge_index=edge_index, batch_lengths=batch_lengths, edge_batch=edge_batch)
                     
class ImgGraphDataset(Dataset):
    def __init__(self, base_root_dir, data_dir, dataset_type: SupportedDatasets, split: Splits, prep_params, graph_params, 
        only_create_dataset, max_out_ram_for_preprocessing, statistics_file_type):
        super().__init__()
        self.root = get_dataset_dir(base_root_dir, dataset_type)
        self.raw_dir = os.path.join(self.root, 'raw')
        if data_dir == '':
            self.data_dir = self.root
        else:
            self.data_dir = data_dir
        self.split=split
        self.dataset_type = dataset_type
        self.dataset = None
        self.processed_file_names_length = None
        self.prep_params = prep_params
        self.graph_params = graph_params
        self.statistics_file_type = statistics_file_type if statistics_file_type is not None else self.split.name
        self.processed_dir = os.path.join(self.root,"processed", self.get_processed_name())
        self.preprocessed_file_path = os.path.join(self.processed_dir,self.split.name)
        self.statistics_file_path = self.preprocessed_file_path.replace(self.split.name, self.statistics_file_type)
        self.max_out_ram_for_preprocessing = max_out_ram_for_preprocessing
        if dataset_type is SupportedDatasets.CIFAR10:
            self.classes = torch.arange(10)
        elif dataset_type is SupportedDatasets.IMAGENET:
            self.classes = torch.arange(1000)
        if not is_preprocessed_data_available(root=self.raw_dir, processed_dir=self.processed_dir, dataset_type=dataset_type, split=split):
            make_dataset_available_for_preprocessing(root=self.raw_dir, data_dir=self.data_dir, processed_dir=self.processed_dir, dataset_type=dataset_type, split=split)
            self.preprocess_data()
        if not only_create_dataset:
            self.load_preprocessed_data()

    def get_processed_name(self):
        name = f'data_{self.split.name}'
        if self.prep_params['use_slic']:
            name += '_slic'
        if self.prep_params['scale_params_to_img']:
            name += '_scaled'
        if self.prep_params['scale_only_min_size']:
            name += '_scaled_only_min_size'
        if self.prep_params['use_slic']:
            name += f'_{self.prep_params["compactness"]}_{self.prep_params["region_size"]}'
        else:
            name += f'_{self.prep_params["sigma"]}_{self.prep_params["scale"]}_{self.prep_params["min_size"]}_{self.prep_params["approx_epsilon"]}'
        return name

    def preprocess_data(self):
        mapping = 'identity_mapping' if self.dataset_type is SupportedDatasets.CIFAR10 else os.path.join(self.processed_dir,"idx_to_name_mapping.txt")
        if self.dataset_type is SupportedDatasets.IMAGENET and self.split.name == 'test':
            used_split_name = 'val'
        else:
            used_split_name = self.split.name
        used_raw_dir = os.path.join(self.raw_dir, used_split_name) if self.dataset_type is SupportedDatasets.CIFAR10 else os.path.join(self.data_dir, used_split_name)
        max_ram_for_data = -1
        # heuristic for determining how much RAM to require for the dataset in the preprocessing
        # the heuristic was done for 96 GB RAM available and the Felzenszwalb hyperparameters presented in the paper
        if self.max_out_ram_for_preprocessing:
            max_ram_for_data = psutil.virtual_memory()[1]//(1024*1024) - 30*1024;
        cpp_args = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'cpp_preprocessing', 'image_to_graph_dataset'),
        used_raw_dir, 
        self.processed_dir,
        self.split.name,
        mapping,
        str(self.prep_params['sigma']), 
        str(self.prep_params['scale']),
        str(self.prep_params['min_size']), 
        str(self.prep_params['approx_epsilon']), 
        str(int(self.prep_params['use_slic'])),
        str(int(self.prep_params['region_size'])),
        str(int(self.prep_params['compactness'])),
        str(int(self.prep_params['scale_params_to_img'])),
        str(int(self.prep_params['scale_only_min_size'])),
        str(int(max_ram_for_data))
        ]
        print(f'Running processing with: {cpp_args}')
        subprocess.run(cpp_args)

    def load_preprocessed_data(self):
        self.data = torch.tensor(np.load(self.preprocessed_file_path+'.npy'),dtype=torch.float32)
        self.offsets = torch.tensor(np.load(self.preprocessed_file_path+'_offsets.npy'),dtype=torch.long)
        self.statistics = torch.tensor(np.load(self.statistics_file_path+'_statistics.npy'),dtype=torch.float)
        self.lengths = self.offsets[1:]-self.offsets[:-1]


    def __getitem__(self, given_idxs):
        return get_data(given_idxs=given_idxs, offsets=self.offsets, lengths=self.lengths, statistics = self.statistics, data=self.data, graph_params=self.graph_params)
        
    def __len__(self):
        return self.lengths.shape[0]
