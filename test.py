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
b = torch.tensor([[0.1,0.2],[0.1,0.2],[2,3]]).cuda()
c = torch.tensor([[0.1,0.2],[0.1,0.2]]).cuda()
c = b.view(-1, 1,2)
print(len(b))


#%%
import random
import numpy as np
import torch
def toy_dataset(DATASET='8gaussians', size=256):
    if DATASET == '25gaussians':
        dataset1 = []
        for i in range(20):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    dataset1.append(point)
        dataset1 = np.array(dataset1, dtype='float32')
        np.random.shuffle(dataset1)
        dataset1 /= 2.828  # stdev
    elif DATASET == '8gaussians':
        scale = 2
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        dataset1 = []
        for i in range(size):
            point = np.random.randn(2) * .02
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset1.append(point)
        dataset1 = np.array(dataset1, dtype='float32')
        dataset1 /= 1.414  # stdev


    return dataset1


def generate_image( true_dist):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    N_POINTS = 128
    RANGE = 3

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))

    plt.clf()

    # true_dist = true_dist.cpu().data.numpy()


    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    # plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())

    plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', marker='+')

    plt.show()

a = toy_dataset('8gaussians')

print(c)
#%%
import random
import numpy as np
a = np.random.normal(size=[2,1]).astype('float32')
print(a)

#%%
for i in range(2):
    print(i)