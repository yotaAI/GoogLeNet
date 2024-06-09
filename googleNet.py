import pandas as pd
import cv2
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import Dataset

torch.manual_seed(123)

arc_google = [
    ('conv',64,7,2,3),
    ('max_pool',3,2),
    ('batch_norm'),
    ('conv',192,3,1,'same'),
    ('batch_norm'),
    ('max_pool',3,2),
    ('inception',[64,128,32,32]),
    ('inception',[128,192,96,64]),
    ('max_pool',3,2),
    ('inception',[192,208,48,64]),
    ('inception',[160,224,64,64]),
    ('inception',[128,256,64,64]),
    ('inception',[112,288,64,64]),
    ('inception',[256,320,128,128]),
    ('max_pool',3,2),
    ('inception',[256,320,128,128]),
    ('inception',[384,384,128,128]),
    ('avg_pool',7,1),
]


class Inception(nn.Module):
    def __init__(self,input_channel,output_channels:list):
        super().__init__()
        self.output_channels=output_channels
        self.input_channel = input_channel


        self.conv1 = nn.Conv2d(self.input_channel,self.output_channels[0],kernel_size=1,stride=1,padding='same')
        self.conv3_1 = nn.Conv2d(self.input_channel,self.output_channels[0],kernel_size=1,stride=1,padding='same')
        self.conv3 = nn.Conv2d(self.output_channels[0],self.output_channels[1],kernel_size=3,padding='same')
        self.conv5_1 = nn.Conv2d(self.input_channel,self.output_channels[0],kernel_size=1,stride=1,padding='same')
        self.conv5 = nn.Conv2d(self.output_channels[0],self.output_channels[2],kernel_size=5,padding='same')

        self.pool = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.pool_conv = nn.Conv2d(self.input_channel,self.output_channels[3],kernel_size=1,stride=1,padding='same')
    def forward(self,x):
        conv1 = self.conv1(x)
        conv3 = self.conv3(self.conv3_1(x))
        conv5 = self.conv5(self.conv5_1(x))

        pool = self.pool_conv(self.pool(x))
        x = torch.cat((conv1,conv3,conv5,pool),dim=1)

        return x


class GoogLeNet(nn.Module):
    def __init__(self,in_channel,architecture):
        super().__init__()
        self.in_channel=in_channel
        self.layers = nn.ModuleList()
        in_channel=self.in_channel

        for layer in architecture:
            if layer[0]=='conv':
                l,out,kernel,stride,padding = layer
                self.layers.append(nn.Conv2d(in_channel,out,stride=stride,kernel_size=kernel,padding=padding))
                in_channel=out
            elif layer[0]=='max_pool':
                l,kernel,stride=layer
                self.layers.append(nn.MaxPool2d(kernel_size=kernel,stride=stride,ceil_mode=True))
            elif layer[0]=='avg_pool':
                l,kernel,stride=layer
                self.layers.append(nn.AvgPool2d(kernel_size=kernel,stride=stride,ceil_mode=True))
            elif layer[0]=='batch_norm':
                self.layers.append(nn.BatchNorm2d(in_channel))
            elif layer[0]=='inception':
                l,out= layer
                self.layers.append(Inception(in_channel,out))
                in_channel=sum(out)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1024,1000),
            nn.Softmax(dim=-1),
        )
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return self.fc(x)


if __name__=='__main__':
    google_net = GoogLeNet(3,arc_google)
    print(summary(google_net,(3,224,224)))