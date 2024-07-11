import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import argparse
# from torchsummary import summary

# Homework 1: 
# calculate the input/output size of each layer 
# calculate the total number of patameters

class NetHW(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 8,  3, 1, 1, bias=False)
        self.conv_2 = nn.Conv2d(8, 8,  5, 1, 2, bias=False)
        self.conv_3 = nn.Conv2d(8, 16, 3, 1, 1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(4*4*16, 10, bias=False)

    def forward(self, x):           # x : [B, 3, 32, 32]
        out = self.conv_1(x)        # [B, 3, 32, 32]  -> [B, c, h, w]
        out = self.maxpool(out)     # [B, c, h, w]  -> [B, c, h, w]
        out = self.conv_2(out)      # [B, c, h, w]  -> [B, c, h, w]
        out = self.maxpool(out)     # [B, c, h, w]  -> [B, c, h, w]
        out = self.conv_3(out)      # [B, c, h, w]  -> [B, c, h, w]
        out = self.maxpool(out)     # [B, c, h, w]  -> [B, c, h, w]
        out = self.flatten(out)     # [B, c, h, w]  -> [B, _]
        out = self.fc_1(out)        # [B, _]  -> [B, _]
        return out

# Total number of patameters: 





