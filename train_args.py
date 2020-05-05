#!/usr/bin/env python3
""" train_args.py
train_args.py command-line args.
"""

import argparse

def get_args():
    """
    """

    parser = argparse.ArgumentParser(
        description="This script lets you train and save your model.",
        usage="python3 train.py flowers/train --gpu --learning_rate 0.001 --epochs 11 --gpu --hidden_units 500",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('data_directory', action="store")

    parser.add_argument('--arch',
                        action="store",
                        default="alexnet",
                        dest='arch',
                        type=str,
                        help='Directory to save the model file.',
                        )
    
    parser.add_argument('--save_dir',
                        action="store",
                        default=".",
                        dest='save_dir',
                        type=str,
                        help='Directory to save the model file.',
                        )

    parser.add_argument('--save_name',
                        action="store",
                        default="checkpoint",
                        dest='save_name',
                        type=str,
                        help='Checkpoint filename.',
                        )

    parser.add_argument('--categories_json',
                        action="store",
                        default="cat_to_name.json",
                        dest='categories_json',
                        type=str,
                        help='Path to file containing the categories.',
                        )

    parser.add_argument('--gpu',
                        action="store_true",
                        dest="use_gpu",
                        default=False,
                        help='Use the GPU to train instead of the CPU')

    hp = parser.add_argument_group('hyperparameters')

    hp.add_argument('--learning_rate',
                    action="store",
                    default=0.001,
                    type=float,
                    help='Learning rate')

    hp.add_argument('--hidden_units', '-hu',
                    action="store",
                    dest="hidden_units",
                    default=[4096],
                    type=int,
                    nargs='+',
                    help='Hidden layer units')
    
    hp.add_argument('--epochs',
                    action="store",
                    dest="epochs",
                    default=1,
                    type=int,
                    help='Epochs to train the model for')

    parser.parse_args()
    return parser


def main():
    """
        Main Function
    """
    print(f'Command line argument utility for train.py.\nTry "python train.py -h".')


if __name__ == '__main__':
    main()
"""
 main() is called if script is executed on it's own.
"""