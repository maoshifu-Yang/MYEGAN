import utilis, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import dataloader
import matplotlib.pyplot as plt
import random
import copy
from torch.autograd import grad
from torch.nn import utils

class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, 2),

        )

        # utilis.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)


        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(2, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128,128),
            nn.LeakyReLU(0.2),
            nn.Linear(128,512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, self.output_dim),

        )
        # utilis.initialize_weights(self)

    def forward(self, input):

        x = self.fc(input)

        return x

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
             if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

class WGAN_TOY(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 62
        self.unrolled_step = 10
        self.datasize = 500
        self.lambda_ = 1

        # load dataset
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        data = self.data_loader.__iter__().__next__()[0]

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=1, input_size=self.input_size)
        self.D = discriminator(input_dim=self.datasize, output_dim=1, input_size=self.input_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        utilis.print_network(self.G)
        utilis.print_network(self.D)
        print('-----------------------------------------------')


        # fixed noise
        self.sample_z_ = torch.randn((self.datasize, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()


    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.datasize, 1), torch.zeros(self.datasize, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')


        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()

            data = self.toy_dataset('25gaussians')

            x_ = torch.from_numpy(data)

            z_ = torch.randn((self.datasize,self.z_dim))
            if self.gpu_mode:
                x_, z_ = x_.cuda(), z_.cuda()

                # update D network
            self.D_optimizer.zero_grad()

            D_real = self.D(x_)

            D_real_loss = -torch.mean(D_real)

            G_ = self.G(z_)
            D_fake = self.D(G_)

            D_fake_loss = torch.mean(D_fake)

            alpah = torch.rand((self.datasize),1)
            alpah = alpah.cuda()

            x_hat = alpah * x_ + (1-alpah) * G_.data

            x_hat.requires_grad = True

            pred_hat = self.D(x_hat)

            gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                                         create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient_penalty = self.lambda_ * ((gradients.view(gradients.size()[0], -1).norm(2, 1)-1 ) ** 2).mean()

            D_loss = D_real_loss + D_fake_loss + gradient_penalty
            self.train_hist['D_loss'].append(D_loss.item())

            D_loss.backward()
            self.D_optimizer.step()

            # update G network

            self.G_optimizer.zero_grad()

                # if self.unrolled_step>0:
                #     backup = copy.deepcopy(self.D)
                #     for i in range(self.unrolled_step):
                #         self.d_unrolled_loop(d_gent_input=z_)
            G_ = self.G(z_)
            D_fake = self.D(G_)
            G_loss = -torch.mean(D_fake)
            self.train_hist['G_loss'].append(G_loss.item())

            G_loss.backward()
            self.G_optimizer.step()
            #
            # if self.unrolled_step>0:
            #     self.D.load(backup)
            #     del backup

        #         if ((iter + 1) % 100) == 0:
        #             print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
        #                   ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))
        #
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                if epoch%100 == 0:
                    print('the epoch:',epoch)
                    self.generate_image(data)
        #
        # self.train_hist['total_time'].append(time.time() - start_time)
        # print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
        #       self.epoch, self.train_hist['total_time'][0]))
        # print("Training finish!... save training results")
        #
        self.save()
        # utilis.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
        #                          self.epoch)
        # utilis.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy()
        # else:
        #     samples = samples.data.numpy().transpose(0, 2, 3, 1)
        self.generate_image(samples)

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

    def toy_dataset(self,DATASET='8gaussians', size=256):
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

        elif DATASET=='8gaussians':
            scale =2
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


    def generate_image(self,true_dist):
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

        #true_dist = true_dist.cpu().data.numpy()
        samples = self.G(self.sample_z_)
        print('generate size is',samples.size())
        samples = samples.cpu().data.numpy()

        x = y = np.linspace(-RANGE, RANGE, N_POINTS)
        # plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())

        plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', marker='+')
        plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')
        plt.show()


    def d_unrolled_loop(self,d_gent_input = None):
        self.D_optimizer.zero_grad()
        realdata = self.toy_dataset('25gaussians',size=500)
        data = torch.from_numpy(realdata).cuda()
        d_real_decision = self.D(data)
        target = torch.ones_like(d_real_decision)
        d_real_error = self.BCE_loss(d_real_decision,target)
        if d_gent_input is None:
            d_gent_input = torch.randn((self.datasize, self.z_dim))
        with torch.no_grad():
            d_fake_data = self.G(d_gent_input)
        d_fake_decision = self.D(d_fake_data)
        target_fake = torch.zeros_like(d_fake_decision)
        d_fake_error = self.BCE_loss(d_fake_decision,target_fake)
        d_loss = d_real_error + d_fake_error
        d_loss.backward()
        self.D_optimizer.step()