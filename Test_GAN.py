import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, noise_size, features):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(noise_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, features)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return F.leaky_relu(self.fc3(x))


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(in_channels, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


if __name__ == "__main__":

    iris = datasets.load_iris()
    x = iris.data
    rows, features = x.shape
    device = torch.device("cuda")
    noise_size = 100
    lr = 0.001
    epochs = 20000

    gene = Generator(noise_size, features).to(device)
    disc = Discriminator(features).to(device)
    g_optimizer = optim.Adam(gene.parameters(), lr=lr)
    d_optimizer = optim.Adam(disc.parameters(), lr=lr)
    criterion = nn.BCELoss()
    for i in range(epochs):
        # Discriminator training
        noise = torch.randn((rows, noise_size)).float().to(device)
        fake_data = gene(noise)
        d_optimizer.zero_grad()
        d_real = disc(torch.Tensor(x).to(device))
        real_label = torch.ones_like(d_real)
        loss_d_real = criterion(d_real, real_label)
        d_fake = disc(fake_data)
        fake_label = torch.zeros_like(d_fake)
        loss_d_fake = criterion(d_fake, fake_label)
        loss_d = (loss_d_real + loss_d_fake) / 2
        loss_d.backward(retain_graph=True)
        d_optimizer.step()
        # Generator training
        g_optimizer.zero_grad()
        output = disc(fake_data)
        loss_g = criterion(output, real_label)
        loss_g.backward(retain_graph=True)
        g_optimizer.step()

    # GAN testing
    noise = torch.randn((rows, noise_size)).float().to(device)
    a = gene(noise).cpu().detach().numpy()
    fig = plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c='b', label='real')
    plt.scatter(a[:, 0], a[:, 1], c='r', marker='s', label='fake')
    plt.show()
