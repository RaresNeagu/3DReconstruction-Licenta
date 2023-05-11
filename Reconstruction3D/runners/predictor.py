import os
import random
from logging import Logger

import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from runners.MainRunner import Runner
from models.Reconstruction3D import Reconstruction3D
from utils.mesh import Ellipsoid


class Predictor(Runner):

    def __init__(self, options, logger: Logger):
        super().__init__(options, logger, training=False)

    def init_fn(self, **kwargs):
        self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)
        self.model = Reconstruction3D(self.options.model, self.ellipsoid,
                                  self.options.dataset.camera_f, self.options.dataset.camera_c,
                                  self.options.dataset.mesh_pos)

    def models_dict(self):
        return {'model': self.model}

    def predict_step(self, input_batch):
        self.model.eval()

        # Run inference
        with torch.no_grad():
            images = input_batch['images']
            out = self.model(images)
            self.save_inference_results(input_batch, out)

    def predict(self):
        self.logger.info("Running predictions...")

        predict_data_loader = DataLoader(self.dataset,
                                         batch_size=self.options.test.batch_size,
                                         pin_memory=self.options.pin_memory,
                                         collate_fn=self.dataset_collate_fn)

        for step, batch in enumerate(predict_data_loader):
            self.logger.info("Predicting [%05d/%05d]" % (step * self.options.test.batch_size, len(self.dataset)))

            self.predict_step(batch)

    def save_inference_results(self, inputs, outputs):
        batch_size = inputs["images"].size(0)
        for i in range(batch_size):
            basename, ext = os.path.splitext(inputs["filepath"][i])
            verts = [outputs["pred_coord"][k][i].cpu().numpy() for k in range(3)]
            for k, vert in enumerate(verts):
                meshname = basename + ".%d.obj" % (k + 1)
                vert_v = np.hstack((np.full([vert.shape[0], 1], "v"), vert))
                mesh = np.vstack((vert_v, self.ellipsoid.obj_fmt_faces[k]))
                np.savetxt(meshname, mesh, fmt='%s', delimiter=" ")
