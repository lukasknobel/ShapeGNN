import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn
from torch_sparse import SparseTensor

from src.models.BuildingBlocks import LinearBlock, ConvBlock, ResBlock, PoolBlock


class BaseGNN(nn.Module):
    """
    Superclass of all other GNN classes
    """
    def __init__(self, linear_dropout=0.0, batch_norm_momentum=0.0, act_fn='relu', agg='add',
                 pool='add', residual_type='add', pooling_dim=-1, **kwargs):
        super().__init__()
        print(f'Ignoring the following keyword arguments when creating {self.__class__.__name__} model: {kwargs}')
        if linear_dropout > 0.0:
            self.linear_dropout = nn.Dropout(linear_dropout)
        else:
            self.linear_dropout = None
        self.pooling_dim = pooling_dim
        self.batch_norm_momentum = batch_norm_momentum
        self.residual_type = residual_type
        self.agg = agg

        # identify activation function
        self.act_fn_name = act_fn.lower()
        if self.act_fn_name == 'relu':
            self.act_fn = nn.ReLU()
        elif self.act_fn_name == 'leakyrelu':
            self.act_fn = nn.LeakyReLU()
        elif self.act_fn_name == 'gelu':
            self.act_fn = nn.GELU()
        else:
            raise ValueError(f'Unknown activation function "{self.act_fn_name}"')

        # identify pooling method
        pool = pool.lower()
        self.pool_name = pool
        if self.pool_name == 'add':
            self.pool = geom_nn.global_add_pool
        elif self.pool_name == 'max':
            self.pool = geom_nn.global_max_pool
        elif pool == 'mean':
            self.pool = geom_nn.global_mean_pool
        elif self._set_att_pooling():
            pass
        else:
            raise ValueError(f'Unknown pooling "{self.pool_name}"')

    def _set_att_pooling(self):
        """
        Create an attention pooling layer and save it in an attribute
        """
        if self.pool_name == 'att':
            if self.pooling_dim == -1:
                raise ValueError(f'pre_pooling_dim has to be specified to use "{self.pool_name}"-pooling')
            self.pool = geom_nn.GlobalAttention(
                gate_nn=nn.Linear(self.pooling_dim, 1)
            )
        else:
            return False
        return True

    def update_pooling_dim(self, pooling_dim):
        """
        Update the pooling dimensions
        """
        if self.pooling_dim == pooling_dim:
            print(f'Ignoring update_pooling_dim: old value = new value = {self.pooling_dim}')
        else:
            print(f'Updating pooling_dim: old value = {self.pooling_dim}, new value = {pooling_dim}')
            self.pooling_dim = pooling_dim
            if self._set_att_pooling():
                print(f'Updated pooling layer using new pooling_dim')

    def get_res_block(self, conv_layer_type, conv_lin_layers, residual_type=None):
        """
        Create a residual block
        """
        conv_block = self.get_conv_block(conv_layer_type, conv_lin_layers)
        if residual_type is None:
            residual_type = self.residual_type
        return ResBlock(conv_block, residual_type)

    def get_pool_block(self, conv_layer_type, conv_lin_layers, pool_factor, graph_pool_type):
        """
        Create a pooling block
        """
        conv_block = self.get_conv_block(conv_layer_type, conv_lin_layers)
        return PoolBlock(conv_block, pool_factor, graph_pool_type=graph_pool_type)

    def get_conv_block(self, conv_layer_type, conv_lin_layers):
        """
        Create a graph convolution block
        """
        return ConvBlock(conv_layer_type, self.agg, self.act_fn, conv_lin_layers,
                         self.get_linear_block)

    def get_linear_block(self, lin_layers, final_layer=False):
        """
        Create a linear block
        """
        return LinearBlock(self.batch_norm_momentum, self.linear_dropout, self.act_fn, lin_layers, final_layer=final_layer)


class ShapeGNN(BaseGNN):

    def __init__(self, num_features, num_classes, num_hidden=128, conv_layer_type='GCNConv', num_res_blocks=1, num_pool_blocks=0, pool_factor=3, pool_hidden_multiplier=2,
                 num_linear_layers_mult=1, residual_type='add', linear_dropout=0.1, batch_norm_momentum=0.9, graph_pool_type='mean', ign_position=False, **kwargs):
        super().__init__(linear_dropout=linear_dropout, batch_norm_momentum=batch_norm_momentum,
                         residual_type=residual_type, pooling_dim=2*num_hidden, **kwargs)
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_hidden = num_hidden
        self.ign_position = ign_position

        # create the first residual block
        res1_llayers = [self.num_features, self.num_hidden] + [self.num_hidden for _ in range(num_linear_layers_mult)]
        self.res1 = self.get_res_block(conv_layer_type, res1_llayers)

        # create position encoder if position is not ignored
        self.position_encoder = None
        if not self.ign_position:
            self.position_encoder = nn.Linear(2, self.res1.output_dim)

        # create the following residual blocks
        res_block_list = []
        input_dim = self.res1.output_dim
        for _ in range(num_res_blocks):
            res_llayers = [input_dim for _ in range(num_linear_layers_mult)] + [input_dim, input_dim]
            res_block = self.get_res_block(conv_layer_type, res_llayers)
            res_block_list.append(res_block)
            input_dim = res_block.output_dim
        self.res_blocks = nn.ModuleList(res_block_list)

        # create pooling blocks
        pool_block_list = []
        for _ in range(num_pool_blocks):
            pool_llayers = [input_dim for _ in range(num_linear_layers_mult)] + [input_dim]
            pool_llayers[-1] = int(pool_llayers[-1]*pool_hidden_multiplier)
            pool_block = self.get_pool_block(conv_layer_type, pool_llayers, pool_factor=pool_factor, graph_pool_type=graph_pool_type)
            pool_block_list.append(pool_block)
            input_dim = pool_block.output_dim
        self.pool_blocks = nn.ModuleList(pool_block_list)

        # create the linear block used to transform node features before global pooling
        pre_pool_llayers = [input_dim] + [input_dim for _ in range(num_linear_layers_mult)] + [self.pooling_dim]
        self.pre_pool_linear = self.get_linear_block(pre_pool_llayers)

        # create last linear block
        mlp_llayers = [self.pooling_dim] + [self.num_hidden for _ in range(num_linear_layers_mult)] + [self.num_classes]
        self.mlp = self.get_linear_block(mlp_llayers, final_layer=True)

    def forward(self, x, adj_t: SparseTensor, pos, batch, edge_index, batch_lengths, edge_batch):
        node_feat, adj_t, pos = self.res1(x=x, adj_t=adj_t, pos=pos, batch=batch)
        if self.ign_position:
            x = node_feat
        else:
            node_pos = self.position_encoder(pos)
            x = torch.add(node_feat, node_pos)

        for res_block in self.res_blocks:
            x, adj_t, pos = res_block(x=x, adj_t=adj_t, pos=pos, batch=batch)

        for pool_block in self.pool_blocks:
            x, edge_index, batch, batch_lengths, edge_batch = pool_block(x=x, edge_index=edge_index, batch=batch, batch_lengths=batch_lengths, edge_batch=edge_batch)

        x = self.pre_pool_linear(x)
        x = self.pool(x, batch)
        out = self.mlp(x)
        return out
