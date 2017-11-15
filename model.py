import torch
import torch.nn as nn
import torch.nn.parallel


class Discriminator(nn.Module):
    def __init__(self, gpu):
        super(Discriminator, self).__init__()
        self.gpu = gpu

        layers = nn.Sequential()
        dim = 64
        layers.add_module(f'd.conv.3-{dim}', nn.Conv2d(3, dim, 4, 2, 1, bias=False))
        layers.add_module(f'd.lrelu.{dim}', nn.LeakyReLU(0.2, inplace=True))
        while dim != 512:
            layers.add_module(f'd.conv.{dim}-{dim*2}', nn.Conv2d(dim, dim * 2, 4, 2, 1, bias=False))
            layers.add_module(f'd.conv.batchnorm.{dim*2}', nn.BatchNorm2d(dim * 2))
            layers.add_module(f'd.lrelu.{dim*2}', nn.LeakyReLU(0.2, inplace=True))
            dim *= 2

        assert dim == 512

        layers.add_module(f'd.conv.{dim}-1', nn.Conv2d(dim, 1, 4, 1, 0, bias=False))
        self.main = layers

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.gpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.gpu))
        else:
            output = self.main(input)
        # output = output.mean(0)
        # return output.view(1) # From DCGAN
        return output.view(output.size(0), -1)


class Generator(nn.Module):
    def __init__(self, gpu):
        super(Generator, self).__init__()
        self.gpu = gpu

        layers = nn.Sequential()
        dim = 64
        layers.add_module(f'g.conv.3-{dim}', nn.Conv2d(3, dim, 4, 2, 1, bias=False))
        layers.add_module(f'g.lrelu.{dim}', nn.LeakyReLU(0.2, inplace=True))
        while dim != 512:
            layers.add_module(f'g.conv.{dim}-{dim*2}', nn.Conv2d(dim, dim * 2, 4, 2, 1, bias=False))
            layers.add_module(f'g.conv.batchnorm.{dim*2}', nn.BatchNorm2d(dim * 2))
            layers.add_module(f'g.lrelu.{dim*2}', nn.LeakyReLU(0.2, inplace=True))
            dim *= 2

        assert dim == 512

        while dim != 64:
            layers.add_module(f'g.convt.{dim}-{dim//2}', nn.ConvTranspose2d(dim, dim // 2, 4, 2, 1, bias=False))
            layers.add_module(f'g.convt.batchnorm.{dim//2}', nn.BatchNorm2d(dim // 2))
            layers.add_module(f'g.relu.{dim//2}', nn.ReLU(inplace=True))
            dim //= 2

        layers.add_module(f'g.convt.{dim}-3', nn.ConvTranspose2d(dim, 3, 4, 2, 1, bias=False))
        layers.add_module('g.tanh.3', nn.Tanh())
        self.main = layers

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.gpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.gpu))
        else:
            output = self.main(input)
        return output
