import torch
import torch.nn as nn
import torch.nn.functional as F

import models.VGG16 as vgg16
import models.ResNet50 as resnet50
from models.layers.GraphBottleneck import GraphBottleneck
from models.layers.GraphConv import GraphConv
from models.layers.GraphUnpooling import GraphUnpooling
from models.layers.GraphProjection import GraphProjection


class Reconstruction3D(nn.Module):

    def __init__(self, options, ellipsoid, camera_f, camera_c, mesh_pos):
        super(Reconstruction3D, self).__init__()

        self.hidden_dim = options.hidden_dim
        self.coord_dim = options.coord_dim
        self.last_hidden_dim = options.last_hidden_dim
        self.init_pts = nn.Parameter(ellipsoid.coord, requires_grad=False)
        self.graphconv_activation = options.graphconv_activation

        self.nn_encoder = resnet50.resnet50()
        # self.nn_encoder = vgg16.Encoder()
        self.features_dim = self.nn_encoder.features_dim + self.coord_dim

        self.gcns = nn.ModuleList([
            GraphBottleneck(6, self.features_dim, self.hidden_dim, self.coord_dim,
                            ellipsoid.adj_mat[0], activation=self.graphconv_activation),
            GraphBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.coord_dim,
                            ellipsoid.adj_mat[1], activation=self.graphconv_activation),
            GraphBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.last_hidden_dim,
                            ellipsoid.adj_mat[2], activation=self.graphconv_activation)
        ])

        self.unpooling = nn.ModuleList([
            GraphUnpooling(ellipsoid.unpool_idx[0]),
            GraphUnpooling(ellipsoid.unpool_idx[1])
        ])

        self.projection = GraphProjection(mesh_pos, camera_f, camera_c, bound=options.z_threshold)

        self.graphconv = GraphConv(in_features=self.last_hidden_dim, out_features=self.coord_dim,
                                   adj_mat=ellipsoid.adj_mat[2])

    def forward(self, img):
        batch_size = img.size(0)
        img_feats = self.nn_encoder(img)
        img_shape = self.projection.image_feature_shape(img)

        init_pts = self.init_pts.data.unsqueeze(0).expand(batch_size, -1, -1)

        x = self.projection(img_shape, img_feats, init_pts)
        x1, x_hidden = self.gcns[0](x)

        x1_up = self.unpooling[0](x1)

        x = self.projection(img_shape, img_feats, x1)
        x = self.unpooling[0](torch.cat([x, x_hidden], 2))

        x2, x_hidden = self.gcns[1](x)

        x2_up = self.unpooling[1](x2)

        x = self.projection(img_shape, img_feats, x2)
        x = self.unpooling[1](torch.cat([x, x_hidden], 2))
        x3, _ = self.gcns[2](x)
        if self.graphconv_activation:
            x3 = F.relu(x3)

        x3 = self.graphconv(x3)


        return {
            "pred_coord": [x1, x2, x3],
            "pred_coord_before_deform": [init_pts, x1_up, x2_up],
        }
