# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:10:46 2024

@author: l50040903
"""
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

# Using CPU device be default. 
device=("cpu")

# Default image size for ResNet-50 is 224*224, 
H=224
W=224

train_transforms=v2.Compose([
    v2.RandomResizedCrop(size=(H,W),scale=[0.6,1],ratio=[0.75,1.33],interpolation=torchvision.transforms.InterpolationMode.BICUBIC,antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    ToTensor(),
    v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]), 
    ])

test_transforms=v2.Compose([
    v2.Resize(size=(H,W),interpolation=torchvision.transforms.InterpolationMode.BICUBIC,antialias=True),
    ToTensor(),
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

# Selecting dataset: CIFAR10/Imagenette. 
#trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=train_transforms)
#testset=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=test_transforms)
trainset=torchvision.datasets.Imagenette(root='./data',size="160px",split='train',download=True,transform=train_transforms)
testset=torchvision.datasets.Imagenette(root='./data',size="160px",split='val',download=True,transform=test_transforms)

batch_size=80

# Create data loaders.
train_dataloader=DataLoader(trainset,batch_size=batch_size,shuffle=True)
test_dataloader=DataLoader(testset,batch_size=batch_size,shuffle=False)

# Initialize a pretrained model, download from torchvision official. 
model=torchvision.models.resnet50(num_classes=10)
state_dict50=torch.load("./model/resnet50-0676ba61.pth")
state_dict50.pop("fc.weight")
state_dict50.pop("fc.bias")
model.load_state_dict(state_dict50,strict=False)
for param in model.parameters():
    param.requires_grad_()
model.fc=torch.nn.Linear(2048,10)
model.fc.requires_grad_()

loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),lr=1e-5,weight_decay=5e-4)

# Train, test, and save the model. 
def train(dataloader,model,loss_fn,optimizer):
    model.train()
    loop=tqdm(enumerate(dataloader),total=len(dataloader),leave=True)
    for batch,(X,y) in loop:
        X,y=X.to(device),y.to(device)

        # Compute prediction error
        pred=model(X)
        loss=loss_fn(pred,y)
        batch_acc=(pred.argmax(1)==y).type(torch.float).sum().item()/batch_size

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loop.set_postfix(loss=loss.item(),acc=batch_acc)
            
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
    
epochs=30
for e in range(epochs):
    print(f"Epoch {e+1}\n-------------------------------")
    train(train_dataloader,model,loss_fn,optimizer)
    test(test_dataloader,model,loss_fn)
    torch.save(model.state_dict(),"./model/resnet50-imagenette_lr=1e-5_epochs="+str(e)+".pth")

print("Done!")

