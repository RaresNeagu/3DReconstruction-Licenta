import torch
import torch.nn as nn


class GraphUnpooling(nn.Module):

    def __init__(self, unpool_idx):
        super(GraphUnpooling, self).__init__()
        self.unpool_idx = unpool_idx

        self.in_num = torch.max(unpool_idx).item()
        self.out_num = self.in_num + len(unpool_idx)

    def forward(self, inputs):
        new_features = inputs[:, self.unpool_idx].clone()
        new_vertices = 0.5 * new_features.sum(2)
        output = torch.cat([inputs, new_vertices], 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_num) + ' -> ' \
               + str(self.out_num) + ')'