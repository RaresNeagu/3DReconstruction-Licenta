from logging import Logger

import numpy as np
import torch
from torch.utils.data import DataLoader

from runners.MainRunner import Runner
from models.layers.chamferWrapper import ChamferDist
from models.Reconstruction3D import Reconstruction3D
from utils.averageMeter import AverageMeter
from utils.mesh import Ellipsoid


class Evaluator(Runner):

    def __init__(self, options, logger: Logger):
        super().__init__(options, logger, training=False)

    def init_fn(self):
        self.chamfer = ChamferDist()
        self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)


        self.model = Reconstruction3D(self.options.model, self.ellipsoid,
                                      self.options.dataset.camera_f, self.options.dataset.camera_c,
                                      self.options.dataset.mesh_pos)
        self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus).cuda()

        self.evaluate_step_count = 0
        self.total_step_count = 0

    def models_dict(self):
        return {'model': self.model}

    def evaluate_f1(self, dis_to_pred, dis_to_gt, pred_length, gt_length, thresh):
        recall = np.sum(dis_to_gt < thresh) / gt_length
        prec = np.sum(dis_to_pred < thresh) / pred_length
        return 2 * prec * recall / (prec + recall + 1e-8)

    def evaluate_chamfer_and_f1(self, pred_vertices, gt_points, labels):
        batch_size = pred_vertices.size(0)
        pred_length = pred_vertices.size(1)
        for i in range(batch_size):
            gt_length = gt_points[i].size(0)
            label = labels[i].cpu().item()
            d1, d2, i1, i2 = self.chamfer(pred_vertices[i].unsqueeze(0), gt_points[i].unsqueeze(0))
            d1, d2 = d1.cpu().numpy(), d2.cpu().numpy()
            self.chamfer_distance.update(np.mean(d1) + np.mean(d2))
            self.f1_tau.update(self.evaluate_f1(d1, d2, pred_length, gt_length, 1E-4))
            self.f1_2tau.update(self.evaluate_f1(d1, d2, pred_length, gt_length, 2E-4))

    def evaluate_step(self, input_batch):
        self.model.eval()

        with torch.no_grad():
            images = input_batch['images']

            out = self.model(images)

            pred_vertices = out["pred_coord"][-1]
            gt_points = input_batch["points_orig"]
            if isinstance(gt_points, list):
                gt_points = [pts.cuda() for pts in gt_points]
            self.evaluate_chamfer_and_f1(pred_vertices, gt_points, input_batch["labels"])

        return out

    def evaluate(self):
        self.logger.info("Running evaluations...")

        self.evaluate_step_count = 0

        test_data_loader = DataLoader(self.dataset,
                                      batch_size=self.options.test.batch_size * self.options.num_gpus,
                                      num_workers=self.options.num_workers,
                                      pin_memory=self.options.pin_memory,
                                      shuffle=self.options.test.shuffle,
                                      collate_fn=self.dataset_collate_fn)

        self.chamfer_distance = AverageMeter()
        self.f1_tau = AverageMeter()
        self.f1_2tau = AverageMeter()

        for step, batch in enumerate(test_data_loader):
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            out = self.evaluate_step(batch)

            if self.evaluate_step_count % self.options.test.summary_steps == 0:
                self.evaluate_summaries()

            self.evaluate_step_count += 1
            self.total_step_count += 1

        for key, val in self.get_result_summary().items():
            scalar = val
            if isinstance(val, AverageMeter):
                scalar = val.avg
            self.logger.info("Test [%06d] %s: %.6f" % (self.total_step_count, key, scalar))

    def get_result_summary(self):
        return {
                "cd": self.chamfer_distance,
                "f1_tau": self.f1_tau,
                "f1_2tau": self.f1_2tau,
            }

    def evaluate_summaries(self):
        self.logger.info("Test Step %06d/%06d (%06d) " % (self.evaluate_step_count,
                                                          len(self.dataset) // (
                                                                  self.options.num_gpus * self.options.test.batch_size),
                                                          self.total_step_count,) \
                         + ", ".join([key + " " + (str(val) if isinstance(val, AverageMeter) else "%.6f" % val)
                                      for key, val in self.get_result_summary().items()]))
