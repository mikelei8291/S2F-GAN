import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', default='data', help='Dataset directory')
parser.add_argument('--mode', type=str, default='train', help='Working mode: train | test')
parser.add_argument('--workers', type=int, default=2, help='Number of workers')
parser.add_argument('--batch_size', type=int, default=64, help='Input batch size')
parser.add_argument('--image_size', type=int, default=200, help='Input image size')
parser.add_argument('--epochs', type=int, default=50000, help='Epochs to train')
parser.add_argument('--lrD', type=float, default=0.00005, help='Learning rate for D-net')
parser.add_argument('--lrG', type=float, default=0.00005, help='Learning rate for G-net')
parser.add_argument('--gpu', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('--clamp', type=float, default=0.01, help='W-GAN clamp value')


def getConfig():
    config = parser.parse_args()
    return config
