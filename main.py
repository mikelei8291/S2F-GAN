#!/usr/bin/env python3
import os
import random
import torch
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable

from model import Generator, Discriminator
from config import getConfig
from data_loader import imageLoader

config = getConfig()
torch.manual_seed(random.randint(1, 10000))

if not os.path.exists('./result'):
    os.makedirs('./result')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def toVariable(x):
    return Variable(x.cuda())


netG = Generator(config.gpu)
netG.apply(weights_init)
netD = Discriminator(config.gpu)
netD.apply(weights_init)

input = torch.FloatTensor(config.batch_size, 3, config.image_size, config.image_size)

# optimizer
optimizerD = optim.RMSprop(netD.parameters(), lr=config.lrD)
optimizerG = optim.RMSprop(netG.parameters(), lr=config.lrG)

netD.cuda()
netG.cuda()

input = input.cuda()

data_path = os.path.join(config.data_path, config.mode)
batch_size = config.batch_size
a_dataloader, b_dataloader = imageLoader(data_path, batch_size, config.image_size, config.workers)

# valid_x_A, valid_x_B = toVariable(a_iter.next()), toVariable(b_iter.next())
# utils.save_image(valid_x_A.data, 'result/valid_x_A.png')
# utils.save_image(valid_x_B.data, 'result/valid_x_B.png')

gen_iter = 0
for epoch in range(config.epochs):
    a_iter, b_iter = iter(a_dataloader), iter(b_dataloader)
    # b_iter = iter(b_dataloader)
    i = 0
    while i < len(b_dataloader):
        # Update D network
        for p in netD.parameters():
            p.requires_grad = True

        if gen_iter < 25 or gen_iter % 500 == 0:
            dis_iter = 100
        else:
            dis_iter = 5
        j = 0
        while j < dis_iter and i < len(b_dataloader):
            j += 1
            for p in netD.parameters():
                p.data.clamp_(-config.clamp, config.clamp)

            x_A, x_B = a_iter.next(), b_iter.next()
            i += 1

            # Train with real
            netD.zero_grad()
            x_B = toVariable(x_B)
            batch_size = x_B.size(0)
            errD_real = netD(x_B)
            one = torch.ones(errD_real.size()).cuda()
            errD_real.backward(one)

            # Train with fake
            x_A = toVariable(x_A)
            errD_fake = netD(x_A)
            mone = one * -1
            errD_fake.backward(mone)
            errD = errD_real - errD_fake
            optimizerD.step()

        # Update G network
        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()
        errG = netD(netG(x_A))
        errG.backward(one)
        optimizerG.step()
        gen_iter += 1

        # Print log
        print(f'[{epoch}/{config.epochs}][{i}/{len(b_dataloader)}][{gen_iter}]')
        print(f'Loss_D: {errD.data[0][0]}, Loss_G: {errG.data[0][0]}')
        print(f'Loss_D_real: {errD_real.data[0][0]}, Loss_D_fake: {errD_fake.data[0][0]}')
        if gen_iter % 500 == 0:
            torch.save(netG.state_dict(), f'result/netG_epoch_{epoch}.pth')
            torch.save(netD.state_dict(), f'result/netD_epoch_{epoch}.pth')
            x_B = x_B.mul(0.5).add(0.5)
            utils.save_image(x_B.data, 'result/real_samples.png')
            fake = netG(x_A)
            fake.data = fake.data.mul(0.5).add(0.5)
            utils.save_image(fake.data, f'result/fake_samples_{gen_iter}.png')
