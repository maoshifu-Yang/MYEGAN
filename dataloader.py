from torch.utils.data import DataLoader
import torch.utils.data as Data
from torchvision import datasets, transforms
import numpy as np
import torch
def toy_dataset(DATASET='8gaussians', size=256):
    if DATASET == '25gaussians':
        dataset1 = []
        for i in range(20):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2)*0.05
                    point[0] += 2*x
                    point[1] += 2*y
                    dataset1.append(point)
        dataset1 = np.array(dataset1, dtype='float32')
        np.random.shuffle(dataset1)
        dataset1 /= 2.828 # stdev
    return dataset1


def dataloader(dataset, input_size, batch_size, split='train'):
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    if dataset == 'mnist':
        transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Lambda(lambda  x:x.repeat(3,1,1)), transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])
        data_loader = DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'fashion-mnist':
        data_loader = DataLoader(
            datasets.FashionMNIST('data/fashion-mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'cifar10':
        data_loader = DataLoader(
            datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'svhn':
        data_loader = DataLoader(
            datasets.SVHN('data/svhn', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'stl10':
        data_loader = DataLoader(
            datasets.STL10('data/stl10', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'lsun-bed':
        data_loader = DataLoader(
            datasets.LSUN('data/lsun', classes=['bedroom_train'], transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'toy-25G':
        x = toy_dataset(DATASET='25gaussians',size=256)
        x_d = torch.from_numpy(x)
        data = Data.TensorDataset(x_d)
        data_loader = DataLoader(dataset=data,batch_size=batch_size,shuffle=True)





    return data_loader

