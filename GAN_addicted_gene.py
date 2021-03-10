import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, in_size):
        super(Generator, self).__init__()
        self.dconv1 = nn.ConvTranspose2d(in_size, out_channels=512, kernel_size=4, stride=1, padding=0)
        self.b_norm1 = nn.BatchNorm2d(512)
        self.dconv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.b_norm2 = nn.BatchNorm2d(256)
        self.dconv3 = nn.ConvTranspose2d(256, 128, 11, 3, 0)
        self.b_norm3 = nn.BatchNorm2d(128)
        self.dconv4 = nn.ConvTranspose2d(128, 64, 11, 3, 0)
        self.b_norm4 = nn.BatchNorm2d(64)
        self.dconv5 = nn.ConvTranspose2d(64, 3, 50, 2, 0)

    def forward(self, x):
        x = F.leaky_relu(self.b_norm1(self.dconv1(x)))
        x = F.leaky_relu(self.b_norm2(self.dconv2(x)))
        x = F.leaky_relu(self.b_norm3(self.dconv3(x)))
        x = F.leaky_relu(self.b_norm4(self.dconv4(x)))
        x = F.leaky_relu(self.dconv5(x))
        return torch.tanh(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, 3)
        self.b_norm1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, 3)
        self.b_norm2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 1024, 3)
        self.b_norm3 = nn.BatchNorm2d(1024)
        self.conv4 = nn.Conv2d(1024, 2048, 3)
        self.b_norm4 = nn.BatchNorm2d(2048)
        self.conv5 = nn.Conv2d(2048, 1, 3)

    def forward(self, x):
        x = F.leaky_relu(self.b_norm1(self.conv1(x)))
        x = F.leaky_relu(self.b_norm2(self.conv2(x)))
        x = F.leaky_relu(self.b_norm3(self.conv3(x)))
        x = F.leaky_relu(self.b_norm4(self.conv4(x)))
        x = self.conv5(x)
        return F.sigmoid(x)


if __name__ == "__main__":
    torch.manual_seed(17)
    path = r'C:\Users\tzach\PycharmProjects\DrugTransform\raw images\Meth effects project'
    batch = 44
    transform_ = torchvision.transforms.Compose([torchvision.transforms.Resize((512, 512)),
                                                 torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize((0), (1))])
    image = torchvision.datasets.ImageFolder(path, transform=transform_)
    data = DataLoader(image, batch_size=batch, shuffle=True, num_workers=4)
    samples = next(iter(data))
    #sample_run = samples[0][0]
    for _,batch in enumerate(data):

        sample = data[i][0].detach().permute(1, 2, 0).numpy()
        plt.imshow(sample)