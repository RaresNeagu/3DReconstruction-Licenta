import os
import pprint
from argparse import ArgumentParser

import numpy as np
import yaml
from easydict import EasyDict as edict

from utils.logger import create_logger

options = edict()

options.name = 'Reconstruction3D'
options.num_workers = 2
options.num_gpus = 1
options.pin_memory = True

options.log_dir = "logs"
options.log_level = "info"
options.checkpoint_dir = "checkpoints"
options.checkpoint = "C:/licenta/Reconstruction3D/resnet50/bestResnet.pt"

options.dataset = edict()
options.dataset.name = "shapenet"
options.dataset.subset_train = "train_tf"
options.dataset.subset_eval = "test_tf"
options.dataset.camera_f = [248., 248.]
options.dataset.camera_c = [111.5, 111.5]
options.dataset.mesh_pos = [0., 0., -0.8]
options.dataset.normalization = True

options.dataset.shapenet = edict()
options.dataset.shapenet.num_points = 3000
options.dataset.predict = edict()
options.dataset.predict.folder = "predictions"
options.dataset.predict.path = "predictions/imagePredict.png"

options.model = edict()
options.model.hidden_dim = 192
options.model.last_hidden_dim = 192
options.model.coord_dim = 3
options.model.graphconv_activation = True
options.model.z_threshold = 0

options.loss = edict()
options.loss.weights = edict()
options.loss.weights.normal = 1.6e-4
options.loss.weights.edge = 0.3
options.loss.weights.laplace = 0.5
options.loss.weights.move = 0.1
options.loss.weights.constant = 1.
options.loss.weights.chamfer = [1., 1., 1.]
options.loss.weights.chamfer_opposite = 1.

options.train = edict()
options.train.num_epochs = 50
options.train.batch_size = 4
options.train.summary_steps = 50
options.train.checkpoint_steps = 10000
options.train.test_epochs = 1
options.train.shuffle = True

options.test = edict()
options.test.dataset = []
options.test.summary_steps = 50
options.test.batch_size = 4
options.test.shuffle = False

options.optim = edict()
options.optim.name = "adam"
options.optim.adam = 0.9
options.optim.sgd_momentum = 0.9
options.optim.lr = 5.0e-05
options.optim.wd = 1.0e-06
options.optim.lr_step = [30, 45]
options.optim.lr_factor = 0.1


def _update_dict(full_key, val, d):
    for vk, vv in val.items():
        if vk not in d:
            raise ValueError("{}.{} does not exist in options".format(full_key, vk))
        if isinstance(vv, list):
            d[vk] = np.array(vv)
        elif isinstance(vv, dict):
            _update_dict(full_key + "." + vk, vv, d[vk])
        else:
            d[vk] = vv


def update_options(options_file):
    with open(options_file) as f:
        options_dict = yaml.safe_load(f)
        if "based_on" in options_dict:
            for base_options in options_dict["based_on"]:
                update_options(os.path.join(os.path.dirname(options_file), base_options))
            options_dict.pop("based_on")
        _update_dict("", options_dict, options)


def gen_options(options_file):
    def to_dict(ed):
        ret = dict(ed)
        for k, v in ret.items():
            if isinstance(v, edict):
                ret[k] = to_dict(v)
            elif isinstance(v, np.ndarray):
                ret[k] = v.tolist()
        return ret

    cfg = to_dict(options)

    with open(options_file, 'w') as f:
        yaml.safe_dump(dict(cfg), f, default_flow_style=False)

def reset_options(options, args, phase='train'):
    if hasattr(args, "batch_size") and args.batch_size:
        options.train.batch_size = options.test.batch_size = args.batch_size
    if hasattr(args, "num_epochs") and args.num_epochs:
        options.train.num_epochs = args.num_epochs
    if hasattr(args, "checkpoint") and args.checkpoint:
        options.checkpoint = args.checkpoint
    if hasattr(args, "folder") and args.folder:
        options.dataset.predict.folder = args.folder
    if hasattr(args, "gpus") and args.gpus:
        options.num_gpus = args.gpus
    if hasattr(args, "shuffle") and args.shuffle:
        options.train.shuffle = options.test.shuffle = True

    options.name = 'Reconstruction3D'

    options.log_dir = 'resnet50'
    print('=> creating {}'.format(options.log_dir))
    os.makedirs(options.log_dir, exist_ok=True)

    options.checkpoint_dir = "resnet50"
    print('=> creating {}'.format(options.checkpoint_dir))
    os.makedirs(options.checkpoint_dir, exist_ok=True)

    logger = create_logger(options, phase=phase)
    options_text = pprint.pformat(vars(options))
    logger.info(options_text)

    return logger


if __name__ == "__main__":
    parser = ArgumentParser("Read options and freeze")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    update_options(args.input)
    gen_options(args.output)
