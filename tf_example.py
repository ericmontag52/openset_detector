import argparse
import os

from train import Train
from test import Test


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default='test', help='Name of the folder the data is saved to inside of results')
    parser.add_argument("-e", "--epoch", type=int, default=5000, help='Number of epochs in training')
    parser.add_argument("-b", "--batch-size", type=int, default=256, help='Batch size in training')
    parser.add_argument("-m", "--model", type=str, default='deep_ensembles_models/sess1', help='Name of the model that will be imported for testing')
    parser.add_argument("-s", "--img-size", type=int, default=28, help='Number of pixels across the img')
    parser.add_argument("-l", "--num-label", type=int, default=10, help='Number of classes in the datasets')
    parser.add_argument("-g", "--gpu-fraction", type=float, default=0.5, help='Percentage of the GPU being utilized')
    parser.add_argument("-v", "--validation-ratio", type=float, default=0.1, help='Percentage of the test dataset that will go to validation')
    args = parser.parse_args()
    os.makedirs(f'results/{args.name}', exist_ok=True)
    Train(name=args.name, epoch=args.epoch, batch_size=args.batch_size, model=args.model, img_size=args.img_size, num_label=args.num_label, gpu_fraction=args.gpu_fraction, validation_ratio=args.validation_ratio)
    Test(name=args.name, model=args.model, img_size=args.img_size, num_label=args.num_label, gpu_fraction=args.gpu_fraction)
