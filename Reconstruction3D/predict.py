import argparse
import sys

from runners.predictor import Predictor
from options import update_options, options, reset_options


def parse_args():
    parser = argparse.ArgumentParser(description='Prediction Entrypoint')
    parser.add_argument('--options', help='experiment options file name', required=False, type=str)

    args, rest = parser.parse_known_args()
    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        update_options(args.options)

    parser.add_argument('--batch-size', help='batch size', type=int)
    parser.add_argument('--checkpoint', help='trained model file', type=str, required=False)
    parser.add_argument('--name', required=False, type=str)
    parser.add_argument('--folder', required=False, type=str)

    options.dataset.name += '_demo'

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger = reset_options(options, args, phase='predict')

    predictor = Predictor(options, logger)
    predictor.predict()


if __name__ == "__main__":
    main()