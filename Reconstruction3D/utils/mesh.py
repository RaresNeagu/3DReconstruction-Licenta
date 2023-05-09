import os
import pickle

import numpy as np
import torch
from scipy.sparse import coo_matrix

import config


def torch_sparse_tensor(indices, value, size):
    coo = coo_matrix((value, (indices[:, 0], indices[:, 1])), shape=size)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.tensor(indices, dtype=torch.long)
    v = torch.tensor(values, dtype=torch.float)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, shape)


class Ellipsoid:

    def __init__(self, mesh_pos, file=config.ELLIPSOID_PATH):
        with open(file, "rb") as fp:
            fp_info = pickle.load(fp, encoding='latin1')

        self.coord = torch.tensor(fp_info[0]) - torch.tensor(mesh_pos, dtype=torch.float)


        self.edges, self.laplace_idx = [], []

        for i in range(3):
            self.edges.append(torch.tensor(fp_info[1 + i][1][0], dtype=torch.long))
            self.laplace_idx.append(torch.tensor(fp_info[7][i], dtype=torch.long))


        self.unpool_idx = [torch.tensor(fp_info[4][i], dtype=torch.long) for i in range(2)]

        self.adj_mat = []
        for i in range(1, 4):
            adj_mat = torch_sparse_tensor(*fp_info[i][1])
            self.adj_mat.append(adj_mat)

        ellipsoid_dir = os.path.dirname(file)
        self.faces = []
        self.obj_fmt_faces = []
        for i in range(1, 4):
            face_file = os.path.join(ellipsoid_dir, "face%d.obj" % i)
            faces = np.loadtxt(face_file, dtype='|S32')
            self.obj_fmt_faces.append(faces)
            self.faces.append(torch.tensor(faces[:, 1:].astype(np.int) - 1))