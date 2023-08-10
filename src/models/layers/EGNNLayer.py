import torch
from torch import nn as nn
from torch_geometric import nn as geom_nn
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor


class EGNNLayer(MessagePassing):
    """ E(n)-equivariant Message Passing Layer """

    def __init__(self, node_features, hidden_features, out_features, agg, act_fn, res_con=True, edge_features=0, pos_dim=2, radius=4):
        super().__init__(aggr=agg)
        self.res_con = node_features == out_features and res_con
        if res_con and not self.res_con:
            print(f'Node features ({node_features}) != out features ({out_features}), no residual connection possible')

        self.pos_dim = pos_dim
        self.radius = radius

        self.edge_func = nn.Sequential(nn.Linear(2 * node_features + edge_features + 1, hidden_features),
                                       act_fn,
                                       nn.Linear(hidden_features, hidden_features),
                                       act_fn)

        self.node_func = nn.Sequential(nn.Linear(node_features + hidden_features, hidden_features),
                                       act_fn,
                                       nn.Linear(hidden_features, out_features))

        self.coord_func = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                        act_fn,
                                        nn.Linear(hidden_features, 1))

        nn.init.xavier_uniform_(self.coord_func[-1].weight, gain=0.001)

    def forward(self, x, adj_t: SparseTensor, pos, batch):
        x, pos = self.propagate(adj_t, x=x, pos=pos)
        if self.radius:
            adj_t = geom_nn.radius_graph(pos, self.radius, batch)
        return x, adj_t, pos

    def message(self, x_i, x_j, pos_i, pos_j):
        """ Create messages """
        pos_diff = torch.sum((pos_i-pos_j).pow(2), dim=-1, keepdim=True).sqrt()
        input_ = [x_i, x_j, pos_diff]
        input_ = torch.cat(input_, dim=-1)
        message = self.edge_func(input_)

        pos_message = (pos_i - pos_j) * self.coord_func(message)
        message = torch.cat((message, pos_message), dim=-1)
        return message

    def update(self, message, x, pos):
        """ Update node features and positions """
        node_message, pos_message = message[:, :-self.pos_dim], message[:, -self.pos_dim:]
        # update node features
        input = torch.cat((x, node_message), dim=-1)
        update = self.node_func(input)
        if self.res_con:
            new_node_feat = x + update
        else:
            new_node_feat = update
        # update positions
        pos = pos + pos_message
        return new_node_feat, pos