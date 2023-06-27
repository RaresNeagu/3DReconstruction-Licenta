import torch.nn as nn
import torch.nn.functional as F

from models.layers.GraphConv import GraphConv


class GraphResBlock(nn.Module):

    def __init__(self, in_dim, hidden_dim, adj_mat, activation=None):
        super(GraphResBlock, self).__init__()

        self.conv1 = GraphConv(in_features=in_dim, out_features=hidden_dim, adj_mat=adj_mat)
        self.conv2 = GraphConv(in_features=hidden_dim, out_features=in_dim, adj_mat=adj_mat)
        self.activation = F.relu if activation else None

    def forward(self, inputs):
        x = self.conv1(inputs)
        if self.activation:
            x = self.activation(x)
        x = self.conv2(x)
        if self.activation:
            x = self.activation(x)

        return (inputs + x) * 0.5


class GraphBottleneck(nn.Module):

    def __init__(self, block_num, in_dim, hidden_dim, out_dim, adj_mat, activation=None):
        super(GraphBottleneck, self).__init__()

        resblock_layers = [
            GraphResBlock(in_dim=hidden_dim, hidden_dim=hidden_dim, adj_mat=adj_mat, activation=activation)
            for _ in range(block_num)]
        self.blocks = nn.Sequential(*resblock_layers)
        self.conv1 = GraphConv(in_features=in_dim, out_features=hidden_dim, adj_mat=adj_mat)
        self.conv2 = GraphConv(in_features=hidden_dim, out_features=out_dim, adj_mat=adj_mat)
        self.activation = F.relu if activation else None

    def forward(self, inputs):
        x = self.conv1(inputs)
        if self.activation:
            x = self.activation(x)
        x_hidden = self.blocks(x)
        x_out = self.conv2(x_hidden)

        return x_out, x_hidden
