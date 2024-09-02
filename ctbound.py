# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['text.usetex'] = True


def D(delta,p):
    a=p-delta
    return a*np.log(a/p)+(1-a)*np.log((1-a)/(1-p))

vfunc=np.vectorize(D)

delta=0.02
p=np.arange(0.03,0.5,0.01)
n=1
y=np.exp(-n*vfunc(delta,p))
plt.plot(p,y)
plt.show()


plt.figure(figsize=(5,4),dpi=200)
A=np.pi**2/2
c0=np.arange(2,11,1)
e1=0.01
e1_=0.05
e1__=0.1
m1=A*(c0**2)*np.log(1/e1)
m1_=A*(c0**2)*np.log(1/e1_)
m1__=A*(c0**2)*np.log(1/e1__)
plt.plot(c0,m1,label="$\epsilon_{1}=0.01.$")
plt.plot(c0,m1_,label="$\epsilon_{1}=0.05.$")
plt.plot(c0,m1__,label="$\epsilon_{1}=0.10.$")
plt.xlim(2,10)
plt.ylim(0,2300)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("Code length.",fontsize=15)
plt.xlabel("$K.$",fontsize=15)
plt.legend(fontsize=15)
plt.show()

