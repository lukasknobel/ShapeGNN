import torch
from torch import nn as nn
from torch_geometric import nn as geom_nn
from torch_sparse import SparseTensor

from src.models.layers.EGNNLayer import EGNNLayer


class LinearBlock(nn.Module):
    def __init__(self, batch_norm_momentum, linear_dropout, act_fn, nodes, final_layer=False):
        super().__init__()
        mlp_layers = []
        for idx in range(1, len(nodes)):
            mlp_layers.append(nn.Linear(nodes[idx-1], nodes[idx]))
            if final_layer and idx == len(nodes) - 1:
                break
            if batch_norm_momentum > 0.0:
                mlp_layers.append(nn.BatchNorm1d(nodes[idx], momentum=batch_norm_momentum))
            mlp_layers.append(act_fn)
            if linear_dropout is not None:
                mlp_layers.append(linear_dropout)
        self.mlp_layers = nn.ModuleList(mlp_layers)

    def forward(self, inp):
        for layer in self.mlp_layers:
            inp = layer(inp)
        return inp


class ConvBlock(nn.Module):

    def __init__(self, conv_layer_type: str, agg, act_fn, nodes, get_linear_block, k=20):
        super().__init__()
        self.conv_layer_type = conv_layer_type
        self.mlp_block = None
        self.act_fn = act_fn
        self.k = k
        self.input_dim = nodes[0]
        self.output_dim = nodes[-1]
        if conv_layer_type == 'edgeconv':
            mod_nodes = list(nodes)
            mod_nodes[0] *= 2
            mlp = get_linear_block(mod_nodes)
            conv_layer = geom_nn.EdgeConv(mlp, aggr=agg)
        elif conv_layer_type == 'dynamicedgeconv':
            mod_nodes = list(nodes)
            mod_nodes[0] *= 2
            mlp = get_linear_block(mod_nodes)
            conv_layer = geom_nn.DynamicEdgeConv(nn=mlp, k=self.k, aggr=agg)
        elif conv_layer_type == 'ginconv':
            mlp = get_linear_block(nodes)
            conv_layer = geom_nn.GINConv(mlp, aggr=agg)
        else:
            if len(nodes) > 2:
                self.mlp_block = get_linear_block(nodes[:-1])
            if conv_layer_type == 'gcnconv':
                conv_layer = geom_nn.GCNConv(nodes[-2], nodes[-1])
            elif conv_layer_type == 'gatconv':
                conv_layer = geom_nn.GATConv(nodes[-2], nodes[-1])
            elif conv_layer_type == 'sageconv':
                conv_layer = geom_nn.SAGEConv(nodes[-2], nodes[-1])
            elif conv_layer_type == 'genconv':
                conv_layer = geom_nn.GENConv(nodes[-2], nodes[-1])
            elif conv_layer_type == 'transformerconv':
                conv_layer = geom_nn.TransformerConv(nodes[-2], nodes[-1])
            elif conv_layer_type == 'gatv2conv':
                conv_layer = geom_nn.GATv2Conv(nodes[-2], nodes[-1])
            elif conv_layer_type == 'egnn':
                conv_layer = EGNNLayer(nodes[-2], nodes[-2], nodes[-1], agg, act_fn, radius=False, res_con=False)
            elif conv_layer_type == 'edgeconv-2':
                mod_nodes = [nodes[-2], nodes[-1]]
                mod_nodes[0] *= 2
                mlp = get_linear_block(mod_nodes)
                conv_layer = geom_nn.EdgeConv(mlp, aggr=agg)
            else:
                raise ValueError(f'Unknown conv_layer_type {conv_layer_type}')
        self.conv_layer = conv_layer

    def forward(self, x, adj_t: SparseTensor, pos, batch):
        output = x
        if self.mlp_block is not None:
            output = self.mlp_block(output)
        if self.conv_layer_type == 'egnn':
            output, adj_t, pos = self.conv_layer(output, adj_t, pos, batch)
            adj_t, pos = adj_t.detach(), pos.detach()
        else:
            output = self.conv_layer(output, adj_t)

        output = self.act_fn(output)
        return output, adj_t, pos


class ResBlock(nn.Module):

    def __init__(self, conv_block: ConvBlock, residual_type='add'):
        super().__init__()
        self.block = conv_block
        self.res_projection = None
        self.residual_type = residual_type
        if self.residual_type == 'add':
            self.input_dim = self.block.input_dim
            self.output_dim = self.block.output_dim
            if self.block.input_dim != self.block.output_dim:
                print(f'Adding linear projection to residual connection to map {self.block.input_dim} to {self.block.output_dim} dimensions')
                self.res_projection = nn.Linear(self.block.input_dim, self.block.output_dim)
        elif self.residual_type == 'cat':
            self.input_dim = self.block.input_dim
            self.output_dim = self.block.input_dim+self.block.output_dim
        elif self.residual_type == 'no':
            self.input_dim = self.block.input_dim
            self.output_dim = self.block.output_dim
        else:
            raise ValueError(f'Unknown residual_type: "{residual_type}"')

    def forward(self, x, adj_t: SparseTensor, pos, batch):
        output, adj_t, pos = self.block(x, adj_t, pos, batch)

        if self.residual_type == 'add':
            if self.res_projection is not None:
                output = output + self.res_projection(x)
            else:
                output = output + x
        elif self.residual_type == 'cat':
            output = torch.concat((x, output), dim=-1)
        elif self.residual_type == 'no':
            pass

        return output, adj_t, pos


class PoolBlock(nn.Module):

    def __init__(self, conv_block: ConvBlock, pool_factor, graph_pool_type):
        super().__init__()
        self.block = conv_block
        self.pool_factor = pool_factor
        self.pool_type = graph_pool_type
        if self.pool_type == 'add':
            self.graph_pool = geom_nn.global_add_pool
        elif self.pool_type == 'max':
            self.graph_pool = geom_nn.global_max_pool
        elif self.pool_type == 'mean':
            self.graph_pool = geom_nn.global_mean_pool
        self.input_dim = self.block.input_dim
        self.output_dim = self.block.output_dim

    def get_node_reduction_batch(self, x, batch):
        idxs = torch.arange(x.shape[0], device=batch.device)
        pooled_idxs = torch.div(idxs, self.pool_factor, rounding_mode='floor')+batch
        
        diff = batch[1:]-batch[:-1]
        caps = diff.nonzero(as_tuple=True)[0]+1
        floor_fix = torch.zeros(pooled_idxs.shape[0], dtype=torch.long, device=pooled_idxs.device)
        floor_fix[caps] = pooled_idxs[caps]-pooled_idxs[caps-1]-1
        floor_fix = torch.cumsum(floor_fix,dim=0)

        return pooled_idxs - floor_fix

    def pool_edge_data(self, edge_index, edge_batch):
        pooled_edges = torch.vstack((torch.div(edge_index,self.pool_factor,rounding_mode='floor')+edge_batch,edge_batch))
        unique_pooled_edges = torch.unique(pooled_edges, dim=1, sorted=True)
        first_edge_index = unique_pooled_edges[0]
        
        
        new_edge_batch = unique_pooled_edges[2]

        diff = new_edge_batch[1:]-new_edge_batch[:-1]
        caps = diff.nonzero(as_tuple=True)[0]+1
        floor_fix = torch.zeros(first_edge_index.shape[0], dtype=torch.long, device=first_edge_index.device)
        floor_fix[caps]=first_edge_index[caps]-first_edge_index[caps-1]-1
        floor_fix=torch.cumsum(floor_fix,dim=0)

        new_edge_index = torch.vstack((first_edge_index,unique_pooled_edges[1])) - floor_fix

        return new_edge_index, new_edge_batch

    def get_new_node_batch(self, node_reduction_batch, batch_lengths):
        batch_diff = torch.hstack((torch.tensor(1, device=node_reduction_batch.device),node_reduction_batch[1:]-node_reduction_batch[:-1]))
        cumsum_batch_diff = torch.cumsum(batch_diff, dim=0)
        used_cumsum_lengths = torch.cumsum(batch_lengths,dim=0)-1
        new_lengths = cumsum_batch_diff[used_cumsum_lengths]
        new_lengths = torch.hstack((new_lengths[0],new_lengths[1:]-new_lengths[:-1]))
        return torch.repeat_interleave(torch.arange(new_lengths.shape[0], device=new_lengths.device),new_lengths), new_lengths

    def forward(self, x, edge_index, batch, batch_lengths, edge_batch):
        node_reduction_batch = self.get_node_reduction_batch(x, batch)
        max_new_batch_size = node_reduction_batch[-1]+1
        new_x = self.graph_pool(x, node_reduction_batch, max_new_batch_size)
        new_edge_index, new_edge_batch = self.pool_edge_data(edge_index, edge_batch)
        sparse_size = new_edge_index[0][-1]+1
        adj_t = SparseTensor(row=new_edge_index[0], col=new_edge_index[1], value=None, sparse_sizes=(sparse_size, sparse_size), is_sorted=True, trust_data=True)
        
        new_batch, new_batch_lengths = self.get_new_node_batch(node_reduction_batch=node_reduction_batch, batch_lengths=batch_lengths)

        output, _, _ = self.block(new_x, adj_t, None, None)
        return output, new_edge_index, new_batch, new_batch_lengths, new_edge_batch