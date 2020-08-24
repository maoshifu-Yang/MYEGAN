import matplotlib.pyplot as plt
import numpy as np
import os,sys
import h5py
import torch
import torch.utils.data as Data




def toy_dataset(DATASET='8gaussians', size=256):

    if DATASET == '25gaussians':
        dataset = []
        for i in range(20):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2)*0.05
                    point[0] += 2*x
                    point[1] += 2*y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        dataset /= 2.828 # stdev
    return dataset
def generate_image(true_dist):
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

    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    #plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())

    plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', marker='+')
    #if not FIXED_GENERATOR:
#%%
a = toy_dataset(DATASET='25gaussians',size=512)
b = torch.from_numpy(a)
torch_dataset = Data.TensorDataset(b)
trainloader = Data.DataLoader(dataset=torch_dataset,shuffle=True)

print(a.shape)
# generate_image(a)
# plt.show()
# print(trainloader)