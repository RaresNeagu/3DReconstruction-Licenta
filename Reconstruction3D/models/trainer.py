import torch
from torch.utils.data import DataLoader

from runners.MainRunner import Runner
from runners.evaluator import Evaluator
from models.losses import Reconstruction3DLoss
from models.Reconstruction3D import Reconstruction3D
from utils.averageMeter import AverageMeter
from utils.mesh import Ellipsoid
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

        self.losses = AverageMeter()

        self.evaluators = [Evaluator(self.options, self.logger)]

    def models_dict(self):
        return {'model': self.model}

    def optimizers_dict(self):
        return {'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler}

    def train_step(self, input_batch):
        self.model.train()

        images = input_batch["images"]

        out = self.model(images)

        loss, loss_summary = self.criterion(out, input_batch)
        self.losses.update(loss.detach().cpu().item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def train(self):
        for epoch in range(self.epoch_count, self.options.train.num_epochs):
            self.epoch_count += 1

            train_data_loader = DataLoader(self.dataset,
                                           batch_size=self.options.train.batch_size * self.options.num_gpus,
                                           num_workers=self.options.num_workers,
                                           pin_memory=self.options.pin_memory,
                                           shuffle=self.options.train.shuffle,
                                           collate_fn=self.dataset_collate_fn)

            self.losses.reset()

            for step, batch in enumerate(train_data_loader):
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                self.train_step(batch)
                self.step_count += 1

                if self.step_count % self.options.train.summary_steps == 0:
                    self.train_summaries(batch)

                if self.step_count % self.options.train.checkpoint_steps == 0:
                    self.dump_checkpoint()

            self.dump_checkpoint()

            # if self.epoch_count % self.options.train.test_epochs == 0:
            # self.test()

            self.lr_scheduler.step()

    def train_summaries(self, input_batch):
        self.logger.debug(input_batch["filename"])

        self.logger.info("Epoch %03d, Step %06d/%06d, Time elapsed %s, Loss %.9f (%.9f)" % (
            self.epoch_count, self.step_count,
            self.options.train.num_epochs * len(self.dataset) // (
                    self.options.train.batch_size * self.options.num_gpus),
            self.time_elapsed, self.losses.val, self.losses.avg))

    def test(self):
        for evaluator in self.evaluators:
            evaluator.evaluate()
