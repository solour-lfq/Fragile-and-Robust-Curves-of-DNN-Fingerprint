# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 08:50:18 2024

@author: l50040903
"""
import copy
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import torch.nn.utils.prune as prune
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import torchattacks
from torchattacks import PGD
"""
# Incorporate if applying quantization as an attack. 
from torch.ao.quantization.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e
)
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
"""

import numpy as np

class DullDataset(Dataset):
    def __init__(self,data,transform=None,label=0):
        self.data=data
        self.label=label
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        if type(self.data[index])==np.ndarray:
            img=torch.from_numpy(self.data[index])
        else:
            img=self.data[index]
        label=self.label
        return img,label

# Using CPU device be default. 
device=("cpu")

# Default image size for ResNet-50 is 224*224, 
H=224
W=224

batch_size=20

loss_fn=nn.CrossEntropyLoss()
dloss_fn=nn.L1Loss()

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

def test(dataloader,model,loss_fn):
    size=len(dataloader.dataset)
    num_batches=len(dataloader)
    #model.eval()
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

# Train all layers.
def train(dataloader,model,loss_fn):
    local_model=copy.deepcopy(model)
    local_model.train()
    for param in local_model.parameters():
        param.requires_grad=True
    optimizer=torch.optim.Adam(filter(lambda p:p.requires_grad,local_model.parameters()),lr=1e-5,weight_decay=5e-4)
    loop=tqdm(enumerate(dataloader),total=len(dataloader),leave=True)
    for batch,(X,y) in loop:
        X,y=X.to(device),y.to(device)
        # Compute prediction error
        pred=local_model(X)
        loss=loss_fn(pred,y)
        batch_acc=(pred.argmax(1)==y).type(torch.float).sum().item()/dataloader.batch_size
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loop.set_postfix(loss=loss.item(),acc=batch_acc)
    return local_model

# Train the last layer. 
def trainll(dataloader,model,loss_fn):
    local_model=copy.deepcopy(model)
    local_model.train()
    for param in local_model.parameters():
        param.requires_grad=False
    local_model.fc.requires_grad_()
    optimizer=torch.optim.Adam(filter(lambda p:p.requires_grad,local_model.parameters()),lr=1e-5,weight_decay=5e-4)
    loop=tqdm(enumerate(dataloader),total=len(dataloader),leave=True)
    for batch,(X,y) in loop:
        X,y=X.to(device),y.to(device)
        # Compute prediction error
        pred=local_model(X)
        loss=loss_fn(pred,y)
        batch_acc=(pred.argmax(1)==y).type(torch.float).sum().item()/dataloader.batch_size
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loop.set_postfix(loss=loss.item(),acc=batch_acc)
    return local_model

# Compute the functional difference of model1 and model2 on a collection of samples. 
def diff(model1,model2,dataloader):
    size=len(dataloader.dataset)
    #model1.eval()
    #model2.eval()
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
            batch_diffs=float(torch.sum(pred1!=pred2)/dataloader.batch_size)
            loop.set_postfix(diffs=batch_diffs)
    diffs=float(diffs)
    diffs/=size
    print(diffs)
    return diffs


def gen_adv(model,dataloader):
    atk=PGD(model,eps=8/255,alpha=2/225,steps=10,random_start=True)
    atk.set_normalization_used(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    loop=tqdm(enumerate(dataloader),total=len(dataloader),leave=True)
    for idx,(X,y) in loop:
        if idx==0:
            adv_images=atk(X,y)
        else:
            adv_images=torch.cat((adv_images,atk(X,y)))
    return adv_images

# Prune the last layer. 
def prune_fc(model,rate):
    local_model=copy.deepcopy(model)
    m=local_model.fc
    prune.random_unstructured(m,name="weight",amount=rate)
    prune.random_unstructured(m,name="bias",amount=rate)
    prune.remove(m,"weight")
    prune.remove(m,"bias")
    #torch.save(local_model.state_dict(),path)
    return local_model
    
# Prune the BN layers. 
def prune_bn(model,rate):
    local_model=copy.deepcopy(model)
    ms=[]
    for module in local_model.modules():
        if isinstance(module,torchvision.models.resnet.Bottleneck):
            ms.append(module.bn1)
            ms.append(module.bn2)
    for m in ms:
        prune.random_unstructured(m,name="weight",amount=rate)
        prune.random_unstructured(m,name="bias",amount=rate)
        prune.remove(m,"weight")
        prune.remove(m,"bias")
    #torch.save(local_model.state_dict(),path)
    return local_model

"""
# Runtime quantization.
def quanti(model,dataloader):
    example_inputs=(next(iter(dataloader))[0],)
    model.eval()
    model_q=copy.deepcopy(model)
    model_q=capture_pre_autograd_graph(model_q,example_inputs)
    quantizer=XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
    model_q=prepare_pt2e(model_q,quantizer)
    #model_.eval()
    with torch.no_grad():
        print("Calibrating model...")
        for image,target in tqdm(dataloader):
            model_q(image)
    model_q=convert_pt2e(model_q)
    return model_q
"""  
  
# Knowledge distillation. 
def distill(teacher,student,dataloader):
    student.train()
    optimizer=torch.optim.Adam(filter(lambda p:p.requires_grad,student.parameters()),lr=1e-5,weight_decay=5e-4)
    loop=tqdm(enumerate(dataloader),total=len(dataloader),leave=True)
    for batch,(X,y) in loop:
        X,y=X.to(device),y.to(device)
        # Compute prediction error
        pred=student(X)
        answer=teacher(X)
        loss=dloss_fn(pred,answer)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loop.set_postfix(loss=loss.item())     
    student.eval()
    return student

# Handcrafted backdoor. 
def hcbackdoor(model,dataloader,poisondata,delta):
    H=10
    attackedmodel=copy.deepcopy(model)
    X,y=next(iter(dataloader))
    x=attackedmodel.conv1(X)
    x=attackedmodel.bn1(x)
    x=attackedmodel.relu(x)
    x=attackedmodel.maxpool(x)
    x=attackedmodel.layer1(x)
    x=attackedmodel.layer2(x)
    x=attackedmodel.layer3(x)
    x=attackedmodel.layer4(x)
    x=attackedmodel.avgpool(x)
    x=torch.flatten(x,1)
    xp=attackedmodel.conv1(poisondata)
    xp=attackedmodel.bn1(xp)
    xp=attackedmodel.relu(xp)
    xp=attackedmodel.maxpool(xp)
    xp=attackedmodel.layer1(xp)
    xp=attackedmodel.layer2(xp)
    xp=attackedmodel.layer3(xp)
    xp=attackedmodel.layer4(xp)
    xp=attackedmodel.avgpool(xp)
    xp=torch.flatten(xp,1)
    vx=torch.mean(x,dim=0)
    vxp=torch.mean(xp,dim=0)
    dv=vxp-vx
    v,i=torch.topk(dv,H)
    attackedclass=9
    for index in i:
        attackedmodel.fc.weight[attackedclass][i]=attackedmodel.fc.weight[attackedclass][i]+delta
    yp0=attackedmodel(X)
    yp0=yp0.argmax(1)
    y0=model(X)
    y0=y0.argmax(1)
    yp=attackedmodel(poisondata)
    yp=yp.argmax(1)
    y=model(poisondata)
    y=y.argmax(1)
    plt.figure(figsize=(5,4),dpi=200)
    plt.hist([y0,yp0,y,yp],label=["Before HCB, normal data.","After HCB, normal data.","Before HCB, triggers.","After HCB, triggers."])
    plt.legend()
    plt.title("HCBackdoor,H=%i"%(H))
    plt.show()
    return attackedmodel

def param_noise(model,t):
    local_model=copy.deepcopy(model)
    temp=local_model.state_dict()
    for key in temp.keys():
        if ("weight" in key) or ("bias" in key):
            noise=torch.randn(temp[key].shape)*t
            temp[key]=temp[key]+noise
    local_model.load_state_dict(temp)
    return local_model

class RandomSmoothed(nn.Module):
    def __init__(self,model,N,t):
        super().__init__()
        self.model=model
        self.N=N
        self.t=t
    def forward(self,x):
        y=self.model(x)
        for n in range(self.N):
            xs=x+torch.randn(x.shape)*self.t
            y=y+self.model(xs)
        y=y/self.N
        return y

class Capsulated(nn.Module):
    def __init__(self,model,variant):
        super().__init__()
        self.model=model
        self.variant=variant
    def forward(self,x):
        y=self.model(x)
        pi=(y.argmax(1)+1)%10
        variant=self.variant(x)
        variant=variant.argmax(1)
        for n in range(len(x)):
            if variant[n]==1:
                y[n][pi]=y[n][pi]+1000
        return y
    
def DIFFS(model1,model2,loader1,loader2,loader3,loader4):
    d1=diff(model1,model2,loader1)
    d2=diff(model1,model2,loader2)
    d3=diff(model1,model2,loader3)
    d4=diff(model1,model2,loader4)
    return [d1,d2,d3,d4]

# debug
f0=torchvision.models.resnet50(num_classes=10)
state_dict0=torch.load("./model/resnet50-imagenette_lr=1e-5_epochs=20.pth")
f0.load_state_dict(state_dict0,strict=True)
for param in f0.parameters():
    param.requires_grad=False
f0.eval()

trainset=torchvision.datasets.Imagenette(root='./data', size="160px", split='train', download=False, transform=train_transforms)
testset=torchvision.datasets.Imagenette(root='./data', size="160px", split='val', download=False, transform=test_transforms)
#trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=False,transform=train_transforms)
trainsubset=Subset(trainset,range(0,200))
#testset=torchvision.datasets.CIFAR10(root='./data',train=False,download=False,transform=test_transforms)
testsubset=Subset(testset,range(1000,3000))
train_dataloader=DataLoader(trainset,batch_size=batch_size,shuffle=True)
trainsub_dataloader=DataLoader(trainsubset,batch_size=batch_size,shuffle=False)
test_dataloader=DataLoader(testset,batch_size=batch_size,shuffle=False)
testsub_dataloader=DataLoader(testsubset,batch_size=batch_size,shuffle=False)

normaldata=Subset(testset,range(0,1000))
noisedata=np.load("./data/smallnoises.npy")
advdata=np.load("./data/eval_imagenette_adv.npy")
Udata=np.load("./data/smallU.npy")

normal_loader=DataLoader(normaldata,batch_size=batch_size,shuffle=False)
noise_loader=DataLoader(DullDataset(noisedata),batch_size=batch_size,shuffle=False)
adv_loader=DataLoader(DullDataset(advdata),batch_size=batch_size,shuffle=False)
U_loader=DataLoader(DullDataset(Udata),batch_size=batch_size,shuffle=False)
    

poisondata=torch.load("./data/test_stamp_triggers.pt")
poison_loader=DataLoader(DullDataset(poisondata),batch_size=batch_size,shuffle=False)
poisondata2=torch.load("./data/abstract_triggers.pt")
poison_loader2=DataLoader(DullDataset(poisondata2),batch_size=batch_size,shuffle=False)

#trainset=torchvision.datasets.Imagenette(root='./data', size="160px", split='train', download=False, transform=ToTensor())


# Fine-tuning test (all layers).
f_ft=copy.deepcopy(f0)
for i in range(10):
    f_ft.train()
    f_ft=train(trainsub_dataloader,f_ft,loss_fn)
    f_ft.eval()
    l=DIFFS(f0,f_ft,normal_loader,noise_loader,adv_loader,U_loader)
    print("FTAL=%.3f,%f,%f,%f,%f"%(i,l[0],l[1],l[2],l[3]))

# Fine-tuning test (layer layer).
f_ft=copy.deepcopy(f0)
for i in range(10):
    f_ft.train()
    f_ft=trainll(trainsub_dataloader,f_ft,loss_fn)
    f_ft.eval()
    l=DIFFS(f0,f_ft,normal_loader,noise_loader,adv_loader,U_loader)
    print("FTAL=%.3f,%f,%f,%f,%f"%(i,l[0],l[1],l[2],l[3]))

# Neuron-pruning test (FC layer).
for i in range(9):
    f_np=prune_bn(f0,0.02*(i+1))
    f_np.eval()
    l=DIFFS(f0,f_np,normal_loader,noise_loader,adv_loader,U_loader)
    print("NPLL=%.3f,%f,%f,%f,%f"%(0.1*(i+1),l[0],l[1],l[2],l[3]))

# Fine-pruning
for i in np.arange(0.01,0.21,0.01):
    f_npal=prune_bn(f0,i)
    f_npal.eval()
    f_fp=train(trainsub_dataloader,f_npal,loss_fn)
    f_fp.eval()
    print(i)
    l=DIFFS(f0,f_fp,normal_loader,noise_loader,adv_loader,U_loader)
    print("FP-%.3f,%f,%f,%f,%f"%(i,l[0],l[1],l[2],l[3]))

# Distill
student=torchvision.models.resnet50(num_classes=10)
state_dict_s=torch.load("./model/resnet50-0676ba61.pth")
state_dict_s.pop("fc.weight")
state_dict_s.pop("fc.bias")
student.load_state_dict(state_dict_s,strict=False)
for param in student.parameters():
    #param.requires_grad=False
    param.requires_grad_()
student.fc=torch.nn.Linear(2048,10)
student.fc.requires_grad_()
for i in range(70):
    student=distill(f0,student,trainsub_dataloader)
for i in range(10):
    student=distill(f0,student,trainsub_dataloader)
    student.eval()
    #test(normal_loader,student,loss_fn)
    l=DIFFS(f0,student,normal_loader,noise_loader,adv_loader,U_loader)
    print("Distill-%.3f,%f,%f,%f,%f"%(i,l[0],l[1],l[2],l[3])) 

# Poisoning
f_poisoned=copy.deepcopy(f0)
for i in range(10):
    f_poisoned.train()
    f_poisoned=train(poison_loader,f_poisoned,loss_fn)
    f_poisoned.eval()
    l=DIFFS(f0,f_poisoned,normal_loader,noise_loader,adv_loader,U_loader)
    print("Poison-1%.3f,%f,%f,%f,%f"%(i,l[0],l[1],l[2],l[3])) 

f_poisoned=copy.deepcopy(f0)
for i in range(10):
    f_poisoned.train()
    f_poisoned=train(poison_loader2,f_poisoned,loss_fn)
    f_poisoned.eval()
    l=DIFFS(f0,f_poisoned,normal_loader,noise_loader,adv_loader,U_loader)
    print("Poison-2%.3f,%f,%f,%f,%f"%(i,l[0],l[1],l[2],l[3])) 

# HCB.
f_hc1=hcbackdoor(f0,train_dataloader,poisondata,0.04)
f_hc2=hcbackdoor(f0,train_dataloader,poisondata,0.08)
f_hc3=hcbackdoor(f0,train_dataloader,poisondata2,0.04)
f_hc4=hcbackdoor(f0,train_dataloader,poisondata2,0.08)
l=DIFFS(f0,f_hc1,normal_loader,noise_loader,adv_loader,U_loader)
print("HCB-%.3f,%f,%f,%f,%f"%(1,l[0],l[1],l[2],l[3]))
l=DIFFS(f0,f_hc2,normal_loader,noise_loader,adv_loader,U_loader)
print("HCB-%.3f,%f,%f,%f,%f"%(2,l[0],l[1],l[2],l[3]))
l=DIFFS(f0,f_hc3,normal_loader,noise_loader,adv_loader,U_loader)
print("HCB-%.3f,%f,%f,%f,%f"%(3,l[0],l[1],l[2],l[3]))
l=DIFFS(f0,f_hc4,normal_loader,noise_loader,adv_loader,U_loader)
print("HCB-%.3f,%f,%f,%f,%f"%(4,l[0],l[1],l[2],l[3]))


# Random smoothing.
f_rs0=RandomSmoothed(f0,3,0.02)
l=DIFFS(f0,f_rs0,normal_loader,noise_loader,adv_loader,U_loader)
print("%.3f,%f,%f,%f,%f"%(1,l[0],l[1],l[2],l[3]))
f_rs1=RandomSmoothed(f0,3,0.04)
l=DIFFS(f0,f_rs1,normal_loader,noise_loader,adv_loader,U_loader)
print("%.3f,%f,%f,%f,%f"%(1,l[0],l[1],l[2],l[3]))
f_rs2=RandomSmoothed(f0,3,0.06)
l=DIFFS(f0,f_rs2,normal_loader,noise_loader,adv_loader,U_loader)
print("%.3f,%f,%f,%f,%f"%(1,l[0],l[1],l[2],l[3]))
f_rs3=RandomSmoothed(f0,3,0.08)
l=DIFFS(f0,f_rs3,normal_loader,noise_loader,adv_loader,U_loader)
print("%.3f,%f,%f,%f,%f"%(1,l[0],l[1],l[2],l[3]))
f_rs4=RandomSmoothed(f0,3,0.1)
l=DIFFS(f0,f_rs4,normal_loader,noise_loader,adv_loader,U_loader)
print("%.3f,%f,%f,%f,%f"%(1,l[0],l[1],l[2],l[3]))


# Cap prepare.
normal_frag=[]
for i in range(200):
    normal_frag.append(normaldata[i][0])
normal_frag=np.stack(normal_frag)
noise_frag=noisedata[0:200]
adv_frag=advdata[0:200]
U_frag=Udata[0:200]

V10=DullDataset(normal_frag,None,0)
V20=DullDataset(noise_frag,None,0)
V30=DullDataset(adv_frag,None,0)
V40=DullDataset(U_frag,None,0)
V11=DullDataset(normal_frag,None,1)
V21=DullDataset(noise_frag,None,1)
V31=DullDataset(adv_frag,None,1)
V41=DullDataset(U_frag,None,1)


V_1=ConcatDataset([V11,V20,V30,V40])
V_2=ConcatDataset([V10,V21,V30,V40])
V_3=ConcatDataset([V10,V20,V31,V40])
V_4=ConcatDataset([V10,V20,V30,V41])
V_1_loader=DataLoader(V_1,batch_size=100,shuffle=True)
V_2_loader=DataLoader(V_2,batch_size=100,shuffle=True)
V_3_loader=DataLoader(V_3,batch_size=100,shuffle=True)
V_4_loader=DataLoader(V_4,batch_size=100,shuffle=True)

Cap=torchvision.models.resnet18(num_classes=2)
state_dict18=torch.load("./model/resnet18-f37072fd.pth")
state_dict18.pop("fc.weight")
state_dict18.pop("fc.bias")
Cap.load_state_dict(state_dict18,strict=False)
for param in Cap.parameters():
    #param.requires_grad=False
    param.requires_grad_()
Cap.fc=torch.nn.Linear(512,2)
Cap.fc.requires_grad_()

Cap1=copy.deepcopy(Cap)
Cap1.train()
Cap1=train(V_1_loader,Cap1,loss_fn)
Cap1=train(V_1_loader,Cap1,loss_fn)
for i in range(5):
    Cap1=train(V_1_loader,Cap1,loss_fn)
    Cap1.eval()
    torch.save(Cap1.state_dict(),"./model/cap_imgnt_normal_"+str(i)+".pth")
    
Cap2=copy.deepcopy(Cap)
Cap2.train()
Cap2=train(V_2_loader,Cap2,loss_fn)
Cap2=train(V_2_loader,Cap2,loss_fn)
for i in range(5):
    Cap2=train(V_2_loader,Cap2,loss_fn)
    Cap2.eval()
    torch.save(Cap2.state_dict(),"./model/cap_imgnt_noise_"+str(i)+".pth")
    
Cap3=copy.deepcopy(Cap)
Cap3.train()
Cap3=train(V_3_loader,Cap3,loss_fn)
Cap3=train(V_3_loader,Cap3,loss_fn)
for i in range(5):
    Cap3=train(V_3_loader,Cap3,loss_fn)
    Cap3.eval()
    torch.save(Cap3.state_dict(),"./model/cap_imgnt_adv_"+str(i)+".pth")
    
Cap4=copy.deepcopy(Cap)
Cap4.train()
Cap4=train(V_4_loader,Cap4,loss_fn)
Cap4=train(V_4_loader,Cap4,loss_fn)
for i in range(5):
    Cap4=train(V_4_loader,Cap4,loss_fn)
    Cap4.eval()
    torch.save(Cap4.state_dict(),"./model/cap_imgnt_U_"+str(i)+".pth")

# Capsulation.
names=["normal","noise","adv","U"]
for i in range(4):
    for j in range(5):
        path="./model/cap_imgnt_"+names[i]+"_"+str(j)+".pth"
        Cap=torchvision.models.resnet18(num_classes=2)
        Cap.load_state_dict(torch.load(path))
        Cap.eval()
        Cap1=Capsulated(f0,Cap)
        l=DIFFS(f0,Cap1,normal_loader,noise_loader,adv_loader,U_loader)
        print("Cap-%.3f,%f,%f,%f,%f"%(i,l[0],l[1],l[2],l[3]))
