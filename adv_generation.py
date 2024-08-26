# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 08:50:18 2024

@author: l50040903
"""
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import torchattacks
from torchattacks import PGD

# Using CPU device be default. 
device=("cpu")

# Default image size for ResNet-50 is 224*224, 
H=224
W=224

train_transforms=v2.Compose([
    ToTensor(),
    v2.RandomResizedCrop(size=(H,W),scale=[0.6,1],ratio=[0.75,1.33],interpolation=torchvision.transforms.InterpolationMode.BICUBIC,antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]), 
    ])

test_transforms=v2.Compose([
    ToTensor(),
    v2.Resize(size=(H,W),interpolation=torchvision.transforms.InterpolationMode.BICUBIC,antialias=True),
    v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),    
    ])

inverse_transforms=v2.Compose([
    v2.Normalize(mean=[0.,0.,0.],std=[1/0.229,1/0.224,1/0.225]),
    v2.Normalize(mean=[-0.485,-0.456,-0.406],std=[1.,1.,1.]),
    ])

# inline using: visual(t[3*H*W])
def visual(t):
    t=inverse_transforms(t)
    toPIL=torchvision.transforms.ToPILImage()
    pic=toPIL(t)
    return pic

# Select dataset, and produce adversarial samples from a subset of the testset. 
#trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=False,transform=train_transforms)
#testset=torchvision.datasets.CIFAR10(root='./data',train=False,download=False,transform=test_transforms)
trainset=torchvision.datasets.Imagenette(root='./data', size="160px", split='train', download=False, transform=train_transforms)
testset=torchvision.datasets.Imagenette(root='./data', size="160px", split='val', download=False, transform=test_transforms)
testsubset=Subset(testset,range(0,1000))

batch_size=80

# Create data loaders.
train_dataloader=DataLoader(trainset,batch_size=batch_size,shuffle=True)
test_dataloader=DataLoader(testsubset,batch_size=batch_size,shuffle=False)


model1=torchvision.models.resnet50(num_classes=10)
state_dict1=torch.load("./model/resnet50-imagenette_lr=1e-5_epochs=20.pth")
model1.load_state_dict(state_dict1,strict=True)
for param in model1.parameters():
    param.requires_grad=False
loss_fn=nn.CrossEntropyLoss()
model1.eval()

# Producing Adversarial Samples, using torchattack official implementations.
atk=PGD(model1,eps=8/255,alpha=2/225,steps=10,random_start=True)
atk.set_normalization_used(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
loop=tqdm(enumerate(test_dataloader),total=len(test_dataloader),leave=True)
for idx,(X,y) in loop:
    if idx==0:
        adv_images=atk(X,y)
    else:
        adv_images=torch.cat((adv_images,atk(X,y)))

torch.save(adv_images,"./data/eval_imagenette_adv.pt")

