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

device=(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

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

#trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=False,transform=train_transforms)
#testset=torchvision.datasets.CIFAR10(root='./data',train=False,download=False,transform=test_transforms)
trainset=torchvision.datasets.Imagenette(root='./data', size="160px", split='train', download=False, transform=train_transforms)
testset=torchvision.datasets.Imagenette(root='./data', size="160px", split='val', download=False, transform=test_transforms)
testsubset=Subset(testset,range(0,1000))

batch_size=80

# Create data loaders.
train_dataloader=DataLoader(trainset,batch_size=batch_size,shuffle=True)
test_dataloader=DataLoader(testsubset,batch_size=batch_size,shuffle=False)

for X,y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

model1=torchvision.models.resnet50(num_classes=10)
state_dict1=torch.load("./model/resnet50-imagenette_lr=1e-5_epochs=20.pth")
model1.load_state_dict(state_dict1,strict=True)
model2=torchvision.models.resnet50(num_classes=10)
state_dict2=torch.load("./model/resnet50-imagenette_lr=1e-5_epochs=19.pth")
model2.load_state_dict(state_dict2,strict=True)
for param in model1.parameters():
    param.requires_grad=False
for param in model2.parameters():
    param.requires_grad=False
loss_fn=nn.CrossEntropyLoss()
model1.eval()
model2.eval()

def test(dataloader,model,loss_fn):
    size=len(dataloader.dataset)
    num_batches=len(dataloader)
    model.eval()
    test_loss,correct=0,0
    with torch.no_grad():
        for X,y in tqdm(dataloader):
            X,y = X.to(device),y.to(device)
            pred=model(X)
            test_loss+=loss_fn(pred,y).item()
            correct+=(pred.argmax(1)==y).type(torch.float).sum().item()
    test_loss/=num_batches
    correct/=size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
#test(test_dataloader,model,loss_fn)

def diff(model1,model2,dataloader):
    size=len(dataloader.dataset)
    model1.eval()
    model2.eval()
    diffs=0
    loop=tqdm(enumerate(dataloader),total=len(dataloader),leave=True)
    with torch.no_grad():
        for idx,(X,y) in loop:
            X,y=X.to(device),y.to(device)
            pred1=model1(X)
            pred2=model2(X)
            pred1=pred1.argmax(1)
            pred2=pred2.argmax(1)
            diffs+=torch.sum(pred1!=pred2)
            batch_diffs=float(torch.sum(pred1!=pred2)/batch_size)
            loop.set_postfix(diffs=batch_diffs)
            break
    diffs=float(diffs)
    diffs/=size
    print(diffs)
    return diffs

# Producing Adversarial Samples.
atk=PGD(model1,eps=8/255,alpha=2/225,steps=10,random_start=True)
atk.set_normalization_used(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
loop=tqdm(enumerate(test_dataloader),total=len(test_dataloader),leave=True)
for idx,(X,y) in loop:
    if idx==0:
        adv_images=atk(X,y)
    else:
        adv_images=torch.cat((adv_images,atk(X,y)))

torch.save(adv_images,"./data/eval_imagenette_adv.pt")

