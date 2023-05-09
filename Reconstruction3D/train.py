import argparse
import sys

import models.trainer
from options import update_options, options, reset_options

def parse_args():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--options',required=False,type=str)
    args, rest = parser.parse_known_args()
    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        update_options(args.options)


    parser.add_argument('--batch-size', help='batch size', type=int)
    parser.add_argument('--checkpoint', help='checkpoint file', type=str)
    parser.add_argument('--num-epochs', help='number of epochs', type=int)
    parser.add_argument('--version', help='version of task (timestamp by default)', type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger = reset_options(options, args)

    trainer = models.trainer.Trainer(options, logger)
    trainer.train()


if __name__ == "__main__":
    main()