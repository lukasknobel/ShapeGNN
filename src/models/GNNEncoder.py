import torch
from torch import nn

from src.models.GNNModels import BaseGNN

from torch_sparse import SparseTensor


class GraphEncoder(BaseGNN):
    """
    The local GNN used to encode superpixel shapes
    """
    def __init__(self, num_features, latent_dim,  hidden_dim=0, conv_layer_type='gcnconv', num_linear_layers_mult=1,
                 num_res_blocks=1, residual_type='no', **kwargs):
        super().__init__(residual_type=residual_type, pooling_dim=1, **kwargs)
        self.latent_dim = latent_dim
        self.conv_layer_type = conv_layer_type
        if self.conv_layer_type == "egnn":
            num_features = 1
        if hidden_dim == 0:
            self.hidden_dim = self.latent_dim
        else:
            self.hidden_dim = hidden_dim

        res_block_list = []
        input_dim = num_features
        for i in range(num_res_blocks):
            if i == 0:
                res_llayers = [input_dim, self.hidden_dim] + [self.hidden_dim for _ in range(num_linear_layers_mult)]
            else:
                res_llayers = [input_dim, input_dim] + [input_dim for _ in range(num_linear_layers_mult)]
            res_block = self.get_res_block(conv_layer_type, res_llayers)
            res_block_list.append(res_block)
            input_dim = res_block.output_dim
        self.res_blocks = nn.ModuleList(res_block_list)

        self.update_pooling_dim(input_dim)

        mlp_llayers = [input_dim for _ in range(num_linear_layers_mult)] + [input_dim, self.latent_dim]
        self.mlp = self.get_linear_block(mlp_llayers, final_layer=True)

    def forward(self, x, adj_t: SparseTensor, batch):
        pos = x
        if self.conv_layer_type == 'egnn':
            x = torch.norm(x, dim=-1, keepdim=True)

        for res_block in self.res_blocks:
            x, adj_t, pos = res_block(x=x, adj_t=adj_t, pos=pos, batch=batch)
        x = self.pool(x, batch)
        z = self.mlp(x)
        return z
