#!/usr/bin/env python3
""" train_args.py
predict_args.py Command-line arguments for predict.py
"""

import argparse


def get_args():
    """
    Get arguments.
    """

    parser = argparse.ArgumentParser(
        description="Image prediction.",
        usage="python3 predict.py /path/to/image.jpg checkpoint.pt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('path_to_image',
                        help='Path to image file.',
                        action="store")

    parser.add_argument('checkpoint_file',
                        help='Path to checkpoint file.',
                        action="store")

    parser.add_argument('--top_k',
                        action="store",
                        default=5,
                        dest='top_k',
                        type=int,
                        help='Return top-k classes.',
                        )

    parser.add_argument('--category_names',
                        action="store",
                        default="cat_to_name.json",
                        dest='categories_json',
                        type=str,
                        help='Path to mapping file containing classes.',
                        )

    parser.add_argument('--gpu',
                        action="store_true",
                        dest="use_gpu",
                        default=False,
                        help='Use the GPU to train instead of the CPU')

    parser.parse_args()
    return parser


def main():
    """
        Main Function
    """
    print(f'Command-line utility for predict.py.\nTry "python train.py -h".')


if __name__ == '__main__':
    main()