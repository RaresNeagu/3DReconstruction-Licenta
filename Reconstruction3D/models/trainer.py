import time
from datetime import timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from runners.MainRunner import Runner
from runners.evaluator import Evaluator
from models.losses import Reconstruction3DLoss
from models.Reconstruction3D import Reconstruction3D
from utils.averageMeter import AverageMeter
from utils.mesh import Ellipsoid
from utils.tensor import recursive_detach
from logging import Logger


class Trainer(Runner):

    def __init__(self, options, logger: Logger):
        super().__init__(options, logger, training=False)

    def init_fn(self):
        self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)

        self.model = Reconstruction3D(self.options.model, self.ellipsoid,
                                      self.options.dataset.camera_f, self.options.dataset.camera_c,
                                      self.options.dataset.mesh_pos)

        self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus).cuda()

        # Setup a joint optimizer for the 2 models
        if self.options.optim.name == "adam":
            self.optimizer = torch.optim.Adam(
                params=list(self.model.parameters()),
                lr=self.options.optim.lr,
                betas=(self.options.optim.adam, 0.999),
                weight_decay=self.options.optim.wd
            )
        else:
            raise NotImplementedError("Your optimizer is not found")
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.options.optim.lr_step, self.options.optim.lr_factor
        )

        self.criterion = Reconstruction3DLoss(self.options.loss, self.ellipsoid).cuda()

        # Create AverageMeters for losses
        self.losses = AverageMeter()

        # Evaluators
        self.evaluators = [Evaluator(self.options, self.logger)]

    def models_dict(self):
        return {'model': self.model}

    def optimizers_dict(self):
        return {'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler}

    def train_step(self, input_batch):
        self.model.train()

        # Grab data from the batch
        images = input_batch["images"]

        # predict with model
        out = self.model(images)

        # compute loss
        loss, loss_summary = self.criterion(out, input_batch)
        self.losses.update(loss.detach().cpu().item())

        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Pack output arguments to be used for visualization
        return recursive_detach(out), recursive_detach(loss_summary)

    def train(self):
        # Run training for num_epochs epochs
        for epoch in range(self.epoch_count, self.options.train.num_epochs):
            self.epoch_count += 1

            # Create a new data loader for every epoch
            train_data_loader = DataLoader(self.dataset,
                                           batch_size=self.options.train.batch_size * self.options.num_gpus,
                                           num_workers=self.options.num_workers,
                                           pin_memory=self.options.pin_memory,
                                           shuffle=self.options.train.shuffle,
                                           collate_fn=self.dataset_collate_fn)

            # Reset loss
            self.losses.reset()

            # Iterate over all batches in an epoch
            for step, batch in enumerate(train_data_loader):
                # Send input to GPU
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Run training step
                out = self.train_step(batch)

                self.step_count += 1

                # Tensorboard logging every summary_steps steps
                if self.step_count % self.options.train.summary_steps == 0:
                    self.train_summaries(batch, *out)

                # Save checkpoint every checkpoint_steps steps
                if self.step_count % self.options.train.checkpoint_steps == 0:
                    self.dump_checkpoint()

            # save checkpoint after each epoch
            self.dump_checkpoint()

            # Run validation every test_epochs
            #if self.epoch_count % self.options.train.test_epochs == 0:
                  #self.test()

            # lr scheduler step
            self.lr_scheduler.step()

    def train_summaries(self, input_batch, out_summary, loss_summary):
        # Debug info for filenames
        self.logger.debug(input_batch["filename"])

        # Save results to log
        self.logger.info("Epoch %03d, Step %06d/%06d, Time elapsed %s, Loss %.9f (%.9f)" % (
            self.epoch_count, self.step_count,
            self.options.train.num_epochs * len(self.dataset) // (
                        self.options.train.batch_size * self.options.num_gpus),
            self.time_elapsed, self.losses.val, self.losses.avg))

    def test(self):
        for evaluator in self.evaluators:
            evaluator.evaluate()