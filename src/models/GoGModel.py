import torch
from torch import nn
from torch_sparse import SparseTensor

from src.models.GNNEncoder import GraphEncoder


class GoGModel(nn.Module):
    """
    Graph of Graphs model with a local GNN (self.encoder) and a global GNN (self.gnn)
    """
    def __init__(self, boundary_node_feat, latent_dim, gnn_model, enc_hidden_dim=0, gog_batch_norm=True,
                 enc_conv_layer_type='gcnconv', enc_linear_layers_mult=1, enc_num_res_blocks=1, enc_residual_type='no'):
        super().__init__()
        self.encoder = GraphEncoder(boundary_node_feat, latent_dim, hidden_dim=enc_hidden_dim,
                                    act_fn=gnn_model.act_fn_name, linear_dropout=gnn_model.linear_dropout.p if gnn_model.linear_dropout is not None else -1,
                                    batch_norm_momentum=gnn_model.batch_norm_momentum,
                                    conv_layer_type=enc_conv_layer_type, agg=gnn_model.agg,
                                    num_linear_layers_mult=enc_linear_layers_mult, num_res_blocks=enc_num_res_blocks,
                                    residual_type=enc_residual_type, pool=gnn_model.pool_name)
        self.gnn = gnn_model
        self.gog_batch_norm = gog_batch_norm
        self.batch_norm = None

        if self.gog_batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.gnn.num_features, momentum=self.gnn.batch_norm_momentum)

    def forward(self, x, adj_t: SparseTensor, pos, batch, sub_x, sub_adj_t: SparseTensor, sub_batch, edge_index, batch_lengths, edge_batch):
        z = self.encoder(x=sub_x, adj_t=sub_adj_t, batch=sub_batch)
        # add shape encoding to main node features
        x = torch.concat((x, z), dim=1)
        batch_norm = self.batch_norm
        if self.gog_batch_norm:
            x = batch_norm(x)
        return self.gnn(x=x, adj_t=adj_t,
                            pos=pos, batch=batch, edge_index=edge_index, batch_lengths=batch_lengths, edge_batch=edge_batch)
