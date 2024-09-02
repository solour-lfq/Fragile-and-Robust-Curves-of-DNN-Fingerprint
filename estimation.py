# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import ConvexHull
plt.rcParams['text.usetex']=True

class MyInterpolation:
    def __init__(self,Xs,Ys):
        self.Xs=Xs
        self.Ys=Ys
    def __call__(self,x):
        if x==0:
            return 1e-5
        for i in range(len(self.Xs)):
            if x<=self.Xs[i]:
                a=self.Ys[i-1]+(self.Ys[i]-self.Ys[i-1])*(x-self.Xs[i-1])/(self.Xs[i]-self.Xs[i-1])
                if a==0:
                    return 1e-5
                else:
                    return a

def visual(coor,logpath):
    coordir={"Normal":0,"Noise":1,"Adv":2,"Universe":3}
    data={}
    data_i={}
    indices=(coordir[coor[0]],coordir[coor[1]])
    raw=np.loadtxt(logpath)
    data["FTAL."]=raw[0:10,indices]
    data["FTLL."]=raw[10:20,indices]
    data["NPAL."]=raw[29:38,indices]
    data["NPFC."]=raw[20:29,indices]
    data["FP."]=raw[38:58,indices]
    data["Distill."]=raw[58:68,indices]
    data["RS."]=raw[92:97,indices]
    data["Poison-1."]=raw[68:78,indices]
    data["Poison-2."]=raw[78:88,indices]
    data["HCB."]=raw[88:92,indices]
    data["Capsulation."]=raw[97:117,indices]
    
    data_i["FTAL."]=raw[0:10,indices]
    data_i["FTLL."]=raw[10:20,indices]
    #data_i["NPAL."]=raw[29:38,indices]
    data_i["NPFC."]=raw[20:29,indices]
    data_i["FP."]=raw[38:58,indices]
    data_i["Distill."]=raw[58:68,indices]
    data_i["RS."]=raw[92:97,indices]
    data_i["Poison-1."]=raw[68:78,indices]
    data_i["Poison-2."]=raw[78:88,indices]
    data_i["HCB."]=raw[88:92,indices]
    #data_i["Capsulation."]=raw[97:117,indices]
    
    # 添加基准攻击.
    raw=np.array([[0,0],[1,1]])
    raw_i=np.array([[0,0],[1,1]])
    for v in data.values():
        raw=np.append(raw,v,axis=0)
    for v in data_i.values():
        raw_i=np.append(raw_i,v,axis=0)    

    #data=np.append(data,np.vstack([np.arange(0,1.001,0.05),np.arange(0,1.001,0.05)]).T,axis=0)
    plt.figure(figsize=(6,6),dpi=200)
    plt.style.use("bmh")
    s=150
    # 可视化所有攻击. 
    #plt.scatter(raw.T[0],raw.T[1],color="cyan")
    for k in data.keys():
        if k=="FTAL.":
            plt.scatter(data[k].T[0],data[k].T[1],label=k,marker="o",edgecolor="black",color="C0",s=s)
        elif k=="FTLL.":
            plt.scatter(data[k].T[0],data[k].T[1],label=k,marker="o",edgecolor="C0",color="white",s=s)
        elif k=="NPFC.":
            plt.scatter(data[k].T[0],data[k].T[1],label=k,marker="s",edgecolor="C1",color="white",s=s)
        elif k=="NPAL.":
            plt.scatter(data[k].T[0],data[k].T[1],label=k,marker="s",edgecolor="black",color="C1",s=s)
        elif k=="FP.":
            plt.scatter(data[k].T[0],data[k].T[1],label=k,marker="d",edgecolor="black",color="C2",s=s)
        elif k=="Distill.":
            plt.scatter(data[k].T[0],data[k].T[1],label=k,marker="*",edgecolor="black",color="C8",s=s)
        elif k=="RS.":
            plt.scatter(data[k].T[0],data[k].T[1],label=k,marker="P",edgecolor="black",color="C3",s=s)
        elif k=="Poison-1.":
            plt.scatter(data[k].T[0],data[k].T[1],label=k,marker="X",edgecolor="black",color="C4",s=s)
        elif k=="Poison-2.":
            plt.scatter(data[k].T[0],data[k].T[1],label=k,marker="X",edgecolor="black",color="C5",s=s)
        elif k=="HCB.":
            plt.scatter(data[k].T[0],data[k].T[1],label=k,marker="X",edgecolor="black",color="C6",s=s)
        elif k=="Capsulation.":
            plt.scatter(data[k].T[0],data[k].T[1],label=k,marker="X",edgecolor="black",color="white",s=s)
        else:
            plt.scatter(data[k].T[0],data[k].T[1],label=k)
    
    # 凸包化，可视化F与R曲线. 
    hull=ConvexHull(raw)
    v=hull.vertices
    v=raw[v]
    v=np.roll(v,-np.where((v==[0,0]).all(axis=1))[0].item(),axis=0)
    i1=np.where((v==[1,1]).all(axis=1))[0].item()
    v=np.append(v,[v[0]],axis=0)
    plt.plot(v[0:i1+1].T[0],v[0:i1+1].T[1],label="Fragility.",color="red")
    plt.plot(v[i1:len(v)].T[0],v[i1:len(v)].T[1],label="Robustness.",color="blue")
    #plt.scatter(v.T[0],v.T[1],label="Support attacks.",color="green",marker=6)

    
    hull_i=ConvexHull(raw_i)
    v_i=hull_i.vertices
    v_i=raw_i[v_i]
    v_i=np.roll(v_i,-np.where((v_i==[0,0]).all(axis=1))[0].item(),axis=0)
    i1_i=np.where((v_i==[1,1]).all(axis=1))[0].item()
    v_i=np.append(v_i,[v_i[0]],axis=0)
    plt.plot(v_i[0:i1_i+1].T[0],v_i[0:i1_i+1].T[1],color="red",linestyle="--")
    plt.plot(v_i[i1_i:len(v_i)].T[0],v_i[i1_i:len(v_i)].T[1],color="blue",linestyle="--")
    

    # 图例信息.
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],[0.0,0.2,0.4,0.6,0.8,1.0],fontsize=15)
    plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],[0.0,0.2,0.4,0.6,0.8,1.0],fontsize=15)
    plt.xlabel("Deviation on "+coor[0]+".",fontsize=20)
    plt.ylabel("Deviation on "+coor[1]+".",fontsize=20)
    plt.legend(fontsize=16,ncol=2,framealpha=0.7,loc="upper center")
    plt.show()
    
    return MyInterpolation(v[0:i1+1].T[0],v[0:i1+1].T[1]),MyInterpolation(v[i1:len(v)].T[0],v[i1:len(v)].T[1])
  
#logpath="./cifar10_log.txt"
logpath="./imagenette_log.txt"
coor01=("Normal","Noise")
F_01,R_01=visual(coor01,logpath)    
coor02=("Normal","Adv")
F_02,R_02=visual(coor02,logpath)
coor03=("Normal","Universe")
F_03,R_03=visual(coor03,logpath)
coor31=("Universe","Noise")
F_31,R_31=visual(coor31,logpath)
coor32=("Universe","Adv")
F_32,R_32=visual(coor32,logpath)
coor21=("Adv","Noise")
F_21,R_21=visual(coor21,logpath)

F_01=np.vectorize(F_01)
R_01=np.vectorize(R_01)
F_02=np.vectorize(F_02)
R_02=np.vectorize(R_02)
F_03=np.vectorize(F_03)
R_03=np.vectorize(R_03)
F_31=np.vectorize(F_31)
R_31=np.vectorize(R_31)
F_32=np.vectorize(F_32)
R_32=np.vectorize(R_32)
F_21=np.vectorize(F_21)
R_21=np.vectorize(R_21)


xi1=0.01
xi2=0.01
tau=0.25

def N1(e1,e2,F):
    return np.log(xi1)/np.log(1-F(e1))

def N2(e1,e2,R):
    return -np.log(xi2)/(2*tau*tau*(1-R(e2))*(1-R(e2)))

def Na(e1,e2,F,R):
    return np.max([N1(e1,e2,F),N2(e1,e2,R)],axis=0)

def Nb(e1,e2,F,R):
    return N1(e1,e2,F)+N2(e1,e2,R)

def visual3d(Fa,Ra,Fb,Rb):
    plt.style.use("default")
    fig=plt.figure(figsize=(6,4),dpi=200)
    ax=fig.add_subplot(projection='3d')
    #ax.view_init(30, 25)
    #x=np.linspace(0.2,0.3,10)
    #y=np.linspace(0.3,0.5,10)
    x=np.linspace(0.5,0.6,10)
    y=np.linspace(0.6,0.8,10)

    X,Y=np.meshgrid(x,y)
    Za=Na(X,Y,Fa,Ra)
    Zb=Nb(X,Y,Fb,Rb)
    ax.plot_surface(X, Y, Za, label="$\mathcal{T}$=Adv.", cmap="autumn", edgecolor='white',alpha=0.8)
    ax.plot_surface(X, Y, Zb, label="$\mathcal{T}_{1},\mathcal{T}_{2}$=Universe,Adv.", cmap="winter", edgecolor='white',alpha=0.8)
    #ax.plot_surface(X, Y, Zb-Za, label="Difference", cmap="autumn", edgecolor='white',alpha=0.5)

    ax.set_xlabel('$\epsilon_{1}$')
    ax.set_ylabel('$\epsilon_{2}$')
    ax.set_zlabel('$N$')
    ax.zaxis.labelpad=-1
    #ax.set_xticks([0.2,0.225,0.25,0.275,0.3])
    #ax.set_yticks([0.3,0.35,0.4,0.45,0.5])
    ax.set_xticks([0.5,0.525,0.55,0.575,0.6])
    ax.set_yticks([0.6,0.65,0.7,0.75,0.8])
    ax.legend()
    plt.show()
    return Za,Zb

Za,Zb=visual3d(F_02,R_02,F_03,R_02)
