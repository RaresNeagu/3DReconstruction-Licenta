import math

import torch
import torch.nn as nn

from utils.tensor import dot


class GraphConv(nn.Module):

    def __init__(self, in_features, out_features, adj_mat, bias=True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.adj_mat = nn.Parameter(adj_mat, requires_grad=False)
        self.weight = nn.Parameter(torch.zeros((in_features, out_features), dtype=torch.float))
        self.loop_weight = nn.Parameter(torch.zeros((in_features, out_features), dtype=torch.float))
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,), dtype=torch.float))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data)
        nn.init.xavier_uniform_(self.loop_weight.data)

    def forward(self, inputs):
        support = torch.matmul(inputs, self.weight)
        support_loop = torch.matmul(inputs, self.loop_weight)
        output = dot(self.adj_mat, support, True) + support_loop
        if self.bias is not None:
            ret = output + self.bias
        else:
            ret = output
        return ret
