#%%
import torch
a=torch.Tensor([[[1,2,3],[4,5,6]]])
b=torch.Tensor([1,2,3,4,5,6])
alpha = torch.rand((12, 1, 1, 1))

print(alpha)
print(b.view(1,6))

#%%
import numpy as np
import matplotlib.pyplot as plt
w = np.array([0,1])

def generatedata(x):
    w_real = 2
    b_real = 1
    y = w_real * x + b_real
    a = np.array([x,y])
    return a



data = []
for i in range(4):
    data.append([generatedata(i)])

data = np.array(data)
print(data[:,0])
#%%

a = 10
print(a//2)
#%%
import torch
x = torch.tensor([[2,3],[3,4]])

print(x.shape)
#%%
x =  x.view(-1,4)
print(x)
#%%
import torch
a = torch.Tensor([[2,3],[3,4],[4,5]])
print(a)
b = a.reshape(1,6)
print(b)


#%%
import torch
import torch.nn as nn
a = nn.BCELoss()
b = torch.tensor([[0.1,0.2],[0.1,0.2]]).cuda()
c = torch.tensor([[0.1,0.2],[0.1,0.2]]).cuda()

print(a(b,c))