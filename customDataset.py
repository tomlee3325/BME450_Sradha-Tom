# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:54:29 2023

@author: PC
"""
#Daniel Bourke Video Tutorial 
#https://www.youtube.com/watch?v=Z_ikDlimN6A&list=RDCMUCr8O8l5cCX85Oem1d18EezQ&start_radio=1 


#Pytorch Domain Libraries
#TorchVision, TorchText, TorchAudio, TorchRec

import torch
from torch import nn

#Note: Pytorch 1.10.0+ is required

print(torch.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device) #cpu #prefer this to be cuda
#print(!nvidia-smi)



#1.Get data
import requests
import zipfile
from pathlib import Path

#Setup Path to a Data folder
data_path = Path(r"C:\Users\PC\Desktop\School\Purdue Undergraduate\Spring 2023\BME 450\Bourke\data")
image_path = data_path / "pneumonia" #reduced normal+pneumonia

#If the image folder doesnt exist
"""
if image_path.is_dir():
    print(f"{image_path} directory already exists...skip download")
else:
    print(f"{image_path} does not exist, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)
    
"""
#2 Data preparation and data exploration

import os
def walk_through_dir(dir_path): #walks through dir_path returning its contents
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

print(walk_through_dir(image_path))
    
#setup train and testing paths

train_dir = image_path / "train"
test_dir = image_path / "test"

print(train_dir, test_dir)

#Visualize
#1. Get all of the image paths
#2. Pick a random image path using Python's random.choce()
#3. Get the image class name using 'pathlib.Path.parent.stem'
#4. Python PIL, Python imaging library. Python Pillow 
#5. show the image and print metadata. 
print("Hi\n", image_path)
# C:\Users\PC\Desktop\School\Purdue Undergraduate\Spring 2023\BME 450\Bourke\data\pneumonia
import random
from PIL import Image

#Set Seed
random.seed(42)

#Get all image paths
#Time Stamp: 20:27:42
image_path_list = list(image_path.glob("*/*/*.jpeg"))
print("Image path list: \n", image_path_list)

#2
#Pick a random image path
random_image_path = random.choice(image_path_list)
print("Random: \n", random_image_path)

#3. Get Image class from path name
image_class = random_image_path.parent.stem
print(image_class)

#4. Open up the image
img = Image.open(random_image_path)
img.show()

#5. Print Metadata
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.height}")

#Try to visualize and iamge with matplotlib
import numpy as np
import matplotlib.pyplot as plt

#Turn image into array
img_as_array = np.asarray(img)
plt.figure(figsize= (10, 7))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | image_shape: {img_as_array.shape} -> [height, width, color_channels]")
plt.axis(False);
print(img_as_array)
#Turn images into pytorch tensors
#Turn the images into numerical representation of the images
#'torch.utils.data.Dataset'
#dataset and dataloader

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#Transforming data with trochvision.transforms
data_transform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()  #0~1
    ])   #Combine transforms together
#also you can use nn.Sequential

print(data_transform(img))
print(data_transform(img).shape)
print(data_transform(img).dtype)

#Visualizing transformed imgaes
#Transforming and Augmenting Images

def plot_transformed_images(image_paths: list, transform, n=3, seed=None):
    """
    selects random images from a path of images and loads/transforms
    them then plots the original vs the transformed version
    
    """
    if seed: #20:50:52
        random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nSize: {f.size}")
            ax[0].axis(False)
            
            #Transform and plot target image
            transformed_image = transform(f).permute(1, 2, 0) #note we will need to change shape for matplotlib (C, H, W) -> (H, W, C)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed\nShape: {transformed_image.shape}")
            ax[1].axis("off")
            
            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
            
plot_transformed_images(image_paths=image_path_list, 
                        transform =data_transform,
                        n=3,
                        seed=40) #change this to see random images


#Loading all of our images and turning them into tensors with torchvision datasets.ImageFolder
##4. Option 1: Loading Image data using 'ImageFolder
#torchvision Dataset
#We can load image classification data using'torchvision.datasets.ImageFolder'
#21:01:36
from torchvision import datasets
train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform, #transform for the data
                                  target_transform = None) #transform for the label
test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)
print(train_data, test_data)

#Get Class names as list
class_name = train_data.classes
print(class_name)

#Get Class names as dict
class_dict = train_data.class_to_idx
print(class_dict)

#Chekc the lengths of our dataset
print(len(train_data), len(test_data))

#train_data.targets
print(train_data.samples[0])

