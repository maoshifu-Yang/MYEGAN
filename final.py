import os, gzip, torch
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import cv2 as cv

def load_mnist(dataset):
    data_dir = os.path.join("./data", dataset)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY).astype(np.int)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1

    X = X.transpose(0, 3, 1, 2) / 255.
    # y_vec = y_vec.transpose(0, 3, 1, 2)

    X = torch.from_numpy(X).type(torch.FloatTensor)
    y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
    return X, y_vec

def load_celebA(dir, transform, batch_size, shuffle):
    # transform = transforms.Compose([
    #     transforms.CenterCrop(160),
    #     transform.Scale(64)
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])

    # data_dir = 'data/celebA'  # this path depends on your computer
    dset = datasets.ImageFolder(dir, transform)
    data_loader = torch.utils.data.DataLoader(dset, batch_size, shuffle)

    return data_loader


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return cv.imwrite(path,image*255)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e+1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)

def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)


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


import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad

import copy
import matplotlib.pyplot as plt
import random
from torch.nn import utils

import torchvision


# %%


class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh()
        )

        initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size) // 4)
        x = self.deconv(x)
        return x


class discriminator(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(self.input_dim, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            utils.spectral_norm(nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024)),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            utils.spectral_norm(nn.Linear(1024, self.output_dim)),
            nn.Sigmoid(),
        )
        initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)
        return x


class EGAN_WGAN(object):
    def __init__(self, epoch, batch_size, result_dir, dataset, log_dir, gan_type, input_size, save_dir, beta1, beta2,
                 gpu_mode):
        # parameter
        self.epoch = epoch
        self.sample_num = 100
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.result_dir = result_dir
        self.dataset = dataset
        self.log_dir = log_dir
        self.gpu_mode = gpu_mode
        self.model_name = gan_type
        self.input_size = input_size
        self.lrG = 0.0002
        self.lrD = 0.0004
        self.beta1 = beta1
        self.beta2 = beta2

        self.lambda_ = 1

        # 先加上 loss的形式
        self.loss_mode = ['WGAN_GP', 'LS', 'DRAGAN']
        self.candinum = 2

        self.z_dim = 62
        self.lambda_ = 10
        self.n_critic = 2  # the number of iterations of the critic per generator iteration

        # load dataset
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        data = self.data_loader.__iter__().__next__()[0]

        # network init
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lrG, betas=(self.beta1, self.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lrD, betas=(self.beta1, self.beta2))

        self.candis = []
        self.G_mutations = []
        self.opt_candis = []

        self.D_iter = 3

        self.device = torch.device("cuda:0")
        torch.autograd.set_detect_anomaly(True)

        for i in range(self.candinum):
            self.candis.append(copy.deepcopy(self.G.state_dict()))
            self.opt_candis.append(copy.deepcopy(self.G_optimizer.state_dict()))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.MSE_loss = nn.MSELoss().cuda()
            self.binary_crossentropy = nn.CrossEntropyLoss().cuda()
            self.BCE = nn.BCELoss().cuda()

        print('---------- Networks architecture -------------')
        print_network(self.G)
        print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise
        self.sample_z_ = torch.randn((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()

        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            # print('The epoch is ',epoch)
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, _) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break
                z_ = torch.randn((self.batch_size, self.z_dim))
                if self.gpu_mode:
                    x_, z_ = x_.cuda(), z_.cuda()

                for i in range(2):
                    # update G
                    if i == 0:
                        self.Fitness, self.evalimgs, self.sel_mut = self.Evo_G(x_, z_)
                        self.evalimgs = torch.cat(self.evalimgs, dim=0)
                        shuffle_ids = torch.randperm(self.evalimgs.size()[0])

                        self.evalimgs = self.evalimgs[shuffle_ids]
                        # print(self.evalimgs.size(),'evalimgs size ')

                    else:
                        self.set_requires_grad(self.D, True)
                        self.D_optimizer.zero_grad()
                        self.gen_imgs = self.evalimgs[(i - 1) * self.batch_size: i * self.batch_size, :].detach()

                        # print(self.evalimgs.size(),'evalimgs in else size')
                        # print(self.gen_imgs.size(),'gen imgs size')

                        self.real_out = self.D(x_)
                        D_real_loss = -torch.mean(self.real_out)
                        self.fake_out = self.D(self.gen_imgs)
                        D_fake_loss = torch.mean(self.fake_out)

                        alpha = torch.rand((self.batch_size, 1, 1, 1))
                        alpha = alpha.cuda()

                        x_hat = alpha * x_.data + (1 - alpha) * self.gen_imgs.data
                        x_hat.requires_grad = True

                        pred_hat = self.D(x_hat)

                        gradients = \
                        grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
                        gradient_penalty = self.lambda_ * (
                                (gradients.view(gradients.size()[0], -1).norm(2, 1) - 0) ** 2).mean()

                        D_loss = D_fake_loss + D_real_loss + gradient_penalty
                        D_loss.backward()
                        self.D_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print(

                        (epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size)
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch + 1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print('epoch',
              np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0])
        print("Training finish!... save training results")

        self.save()
        generate_animation(
            self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
            self.epoch)
        loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name),
                  self.model_name)

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        # load current best G
        F = self.Fitness[:, 2]
        idx = np.where(F == max(F))[0][0]
        self.G.load_state_dict(self.candis[idx])

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.randn((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def fitness_score(self, eval_fake_imgs, eval_real_imgs):

        self.set_requires_grad(self.D, True)

        eval_fake = self.D(eval_fake_imgs)
        eval_real = self.D(eval_real_imgs)

        fake_loss = torch.mean(eval_fake)
        real_loss = -torch.mean(eval_real)

        D_loss_score = fake_loss + real_loss

        # quality fitness score
        Fq = nn.functional.sigmoid(eval_fake).data.mean().cpu().numpy()

        # Diversity fitness score

        gradients = torch.autograd.grad(outputs=D_loss_score, inputs=self.D.parameters(),
                                        grad_outputs=torch.ones(D_loss_score.size()).to(self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)

        with torch.no_grad():
            for i, grad in enumerate(gradients):
                grad = grad.view(-1)
                allgrad = grad if i == 0 else torch.cat([allgrad, grad])

        Fd = -torch.log(torch.norm(allgrad)).data.cpu().numpy()

        return Fq, Fd

    def G_loss(self, Dfake, loss_type):

        if loss_type == 'LS':
            gloss = self.MSE_loss(Dfake, self.y_real_)

        if loss_type == 'WGAN_GP':
            gloss = -torch.mean(Dfake)

        if loss_type == 'DRAGAN':
            gloss = self.BCE(Dfake, self.y_real_)

        return gloss

    def Evo_G(self, inputimage, z_):

        self.real_out = self.D(inputimage)
        F_list = np.zeros(self.candinum)
        Fit_list = []
        G_list = []
        optG_list = []
        selectmutation = []
        count = 0
        lamba_f = 0.1
        eval_fake_image_list = []

        for i in range(self.candinum):
            for j, criterionG in enumerate(self.loss_mode):

                self.G.load_state_dict(self.candis[i])
                self.G_optimizer.load_state_dict(self.opt_candis[i])
                self.G_optimizer.zero_grad()
                self.gen_imgs = self.G(z_)
                # print(gen_img)
                self.fake_out = self.D(self.gen_imgs)

                g_loss = self.G_loss(self.fake_out, criterionG)
                self.set_requires_grad(self.D, False)
                g_loss.backward()
                self.G_optimizer.step()

                with torch.no_grad():
                    eval_fake_imgs = self.G(z_)

                Fq, Fd = self.fitness_score(eval_fake_imgs, inputimage)

                F = Fq + lamba_f * Fd

                if count < self.candinum:
                    F_list[count] = F  # 关于F的列表
                    Fit_list.append([Fq, Fd, F])  # 把当前的放进去
                    G_list.append(copy.deepcopy(self.G.state_dict()))  # 生成器优化器的参数
                    optG_list.append(copy.deepcopy(self.G_optimizer.state_dict()))  # 生成器网络参数
                    eval_fake_image_list.append(eval_fake_imgs)
                    selectmutation.append(self.loss_mode[j])
                # update
                # candi_num 是生存下来的数量
                else:  # 下一代就选到里面
                    fit_com = F - F_list
                    if max(fit_com) > 0:
                        ids_replace = np.where(fit_com == max(fit_com))[0][0]
                        F_list[ids_replace] = F  #
                        Fit_list[ids_replace] = [Fq, Fd, F]
                        G_list[ids_replace] = copy.deepcopy(self.G.state_dict())
                        optG_list[ids_replace] = copy.deepcopy(self.G_optimizer.state_dict())
                        eval_fake_image_list[ids_replace] = eval_fake_imgs
                        selectmutation[ids_replace] = self.loss_mode[j]

                count = count + 1
        self.candis = copy.deepcopy(G_list)
        self.opt_candis = copy.deepcopy(optG_list)

        return np.array(Fit_list), eval_fake_image_list, selectmutation

    #

    def toy_dataset(self, DATASET='8gaussians', size=256):
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

    def generate_image(self, true_dist):
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
        samples = self.G(self.sample_z_)

        samples = samples.cpu().data.numpy()

        x = y = np.linspace(-RANGE, RANGE, N_POINTS)
        # plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())

        plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', marker='+')
        plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')
        plt.show()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


import argparse, os, torch


def main():



    # if opt is None:
    #     exit()

    # if opt.benchmark_mode:
    #     torch.backends.cudnn.benchmark = True
    gan = EGAN_WGAN(gan_type='EGAN2',dataset = 'mnist', epoch = 50,batch_size = 64,input_size = 28, save_dir = 'models', result_dir = 'results', log_dir = 'logs',beta1 = 0.5,beta2 = 0.999,gpu_mode = True)
        # declare instance for GAN
        # launch the graph in a session

    gan.train()
    print(" [*] Training finished!")

    # visualize learned generator
    # gan.visualize_results(args.epoch)
    print(" [*] Testing finished!")

if __name__ == '__main__':

    main()