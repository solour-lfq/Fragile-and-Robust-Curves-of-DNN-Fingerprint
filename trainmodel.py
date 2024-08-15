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

#trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=False,transform=train_transforms)
#testset=torchvision.datasets.CIFAR10(root='./data',train=False,download=False,transform=test_transforms)
trainset=torchvision.datasets.Imagenette(root='./data', size="160px", split='train', download=False, transform=train_transforms)
testset=torchvision.datasets.Imagenette(root='./data', size="160px", split='val', download=False, transform=test_transforms)

batch_size=80

# Create data loaders.
train_dataloader=DataLoader(trainset,batch_size=batch_size,shuffle=True)
test_dataloader=DataLoader(testset,batch_size=batch_size,shuffle=False)

for X,y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*H*W, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = torch.nn.Linear(16 * 8 * 8, 120)  # 6*6 from image dimension
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化模型
model=torchvision.models.resnet50(num_classes=10)
state_dict50=torch.load("./model/resnet50-0676ba61.pth")
state_dict50.pop("fc.weight")
state_dict50.pop("fc.bias")
model.load_state_dict(state_dict50,strict=False)
for param in model.parameters():
    #param.requires_grad=False
    param.requires_grad_()
model.fc=torch.nn.Linear(2048,10)
model.fc.requires_grad_()

#model.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
#model.fc=torch.nn.Linear(2048,10)

loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),lr=1e-5,weight_decay=5e-4)

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
    
epochs=26
for e in range(epochs):
    print(f"Epoch {e+1}\n-------------------------------")
    train(train_dataloader,model,loss_fn,optimizer)
    test(test_dataloader,model,loss_fn)
    torch.save(model.state_dict(),"./model/resnet50-imagenette_lr=1e-5_epochs="+str(e)+".pth")

print("Done!")

