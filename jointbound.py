# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 19:51:32 2024

@author: l50040903
"""

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

def F(e1):
    return e1

def R(e2):
    return e2

xi1=0.1
xi2=0.1
tau=0.22

def N1(e1,e2):
    return np.log(xi1)/np.log(1-F(e1))

def N2(e1,e2):
    return -np.log(xi2)/(2*tau*tau*(1-R(e2))*(1-R(e2)))

def Na(e1,e2):
    return np.max([N1(e1,e2),N2(e1,e2)],axis=0)

def Nb(e1,e2):
    return N1(e1,e2)+N2(e1,e2)

fig = plt.figure(figsize=(5,4),dpi=200)
ax = plt.axes(projection='3d')
#ax.view_init(30, 25)
x = np.linspace(0.05,0.2,15)
y = np.linspace(0.05,0.2,15)

X, Y = np.meshgrid(x, y)
Za=Na(X,Y)
Zb = Nb(X, Y)
ax.plot_surface(X, Y, Za, cmap="winter", edgecolor='white',alpha=0.5)
ax.plot_surface(X, Y, Zb, cmap="autumn", edgecolor='white',alpha=0.5)
ax.set_xlabel('e1')
ax.set_ylabel('e2')
ax.set_zlabel('N')
ax.zaxis.labelpad=-4
ax.set_xticks([0.05,0.1,0.15,0.2])
ax.set_yticks([0.05,0.1,0.15,0.2])
#ax.set_zticks([20,30,40])

"""
# Data for a three-dimensional line
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');
"""


"""
def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
"""