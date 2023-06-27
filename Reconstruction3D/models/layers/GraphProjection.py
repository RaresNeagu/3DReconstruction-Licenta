import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Threshold


class GraphProjection(nn.Module):

    def __init__(self, mesh_pos, camera_f, camera_c, bound=0):
        super(GraphProjection, self).__init__()
        self.mesh_pos, self.camera_f, self.camera_c = mesh_pos, camera_f, camera_c
        self.threshold = None
        self.bound = 0
        if self.bound != 0:
            self.threshold = Threshold(bound, bound)

    def bound_val(self, x):
        if self.bound < 0:
            return -self.threshold(-x)
        elif self.bound > 0:
            return self.threshold(x)
        return x

    @staticmethod
    def image_feature_shape(img):
        return np.array([img.size(-1), img.size(-2)])

    def forward(self, resolution, img_features, inputs):
        half_resolution = (resolution - 1) / 2
        camera_c_offset = np.array(self.camera_c) - half_resolution
        positions = inputs + torch.tensor(self.mesh_pos, device=inputs.device, dtype=torch.float)
        w = -self.camera_f[0] * (positions[:, :, 0] / self.bound_val(positions[:, :, 2])) + camera_c_offset[0]
        h = self.camera_f[1] * (positions[:, :, 1] / self.bound_val(positions[:, :, 2])) + camera_c_offset[1]

        w /= half_resolution[0]
        h /= half_resolution[1]

        w = torch.clamp(w, min=-1, max=1)
        h = torch.clamp(h, min=-1, max=1)

        feats = [inputs]
        for img_feature in img_features:
            feats.append(self.project( img_feature, torch.stack([w, h], dim=-1)))

        output = torch.cat(feats, 2)

        return output

    def project(self, img_feat, sample_points):
        output = F.grid_sample(img_feat, sample_points.unsqueeze(1), align_corners=True)
        output = torch.transpose(output.squeeze(2), 1, 2)

        return output
