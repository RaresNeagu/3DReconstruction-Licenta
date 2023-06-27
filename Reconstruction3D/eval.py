import argparse
import sys

from runners.evaluator import Evaluator
from options import update_options, options, reset_options

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--options',required=False,type=str)

    args, rest = parser.parse_known_args()
    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        update_options(args.options)

    parser.add_argument('--checkpoint', help='checkpoint file', type=str)
    parser.add_argument('--name', help='name', type=str)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    logger = reset_options(options, args, phase='test')

    evaluator = Evaluator(options, logger)
    evaluator.evaluate()

if __name__ == "__main__":
    main()