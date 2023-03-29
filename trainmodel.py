# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 08:56:08 2023

@author: PC
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms  #https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48 
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.io import read_image
import torch.optim as optim
from torch.utils.data import DataLoader #help create mini batches
import customData
from customData import pneumoniaOrnot

#Hyperparameters

batch_size = 3

#Loading the dataset
dataset = pneumoniaOrnot(csv_file = 'pneumoNormal.csv', root_dir = 'pneumonia', transform = transforms.ToTensor()) #Combined dataset
#Creating batch
train_set, test_set = torch.utils.data.random_split(dataset, [20000, 5000])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)