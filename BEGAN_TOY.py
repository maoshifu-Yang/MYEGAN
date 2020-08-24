import utilis, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import dataloader
import matplotlib.pyplot as plt

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
            nn.Linear(1024, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
        )

        utilis.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)

        return x

class discriminator(nn.Module):
    # It must be Auto-Encoder style architecture
    # Architecture : (64)4c2s-FC32-FC64*14*14_BR-(1)4dc2s_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size


        self.fc = nn.Sequential(
            nn.Linear(500, 32),
            nn.Linear(32, 500),

        )

        utilis.initialize_weights(self)

    def forward(self, input):
        input = input.permute(1,0)
        x = self.fc(input)


        return x

class BEGAN_TOY(object):
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
        self.gamma = 1
        self.lambda_ = 0.001
        self.k = 0.0
        self.lr_lower_boundary = 0.00002

        # load dataset
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        data = self.data_loader.__iter__().__next__()[0]


        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=0.0002, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=0.0002, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()

        print('---------- Networks architecture -------------')
        utilis.print_network(self.G)
        utilis.print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        self.sample_z_ = self.sample_z_.cuda()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.M = {}
        self.M['pre'] = []
        self.M['pre'].append(1)
        self.M['cur'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            # for iter, (x_, _) in enumerate(self.data_loader):
            for iter, (x_, ) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))

                if self.gpu_mode:
                    x_, z_ = x_.cuda(), z_.cuda()



                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_)
                D_real = D_real.permute(1,0)
                D_real_loss = torch.mean(torch.abs(D_real - x_))

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = torch.mean(torch.abs(D_fake - G_))

                D_loss = D_real_loss - self.k * D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = torch.mean(torch.abs(D_fake - G_))

                G_loss = D_fake_loss
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()

                # convergence metric
                temp_M = D_real_loss + torch.abs(self.gamma * D_real_loss - G_loss)

                # operation for updating k
                temp_k = self.k + self.lambda_ * (self.gamma * D_real_loss - G_loss)
                temp_k = temp_k.item()

                self.k = min(max(temp_k, 0), 1)
                self.M['cur'] = temp_M.item()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, M: %.8f, k: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item(), self.M['cur'], self.k))


            # if epoch == 0:
            #     self.M['pre'] = self.M['cur']
            #     self.M['cur'] = []
            # else:
            if np.mean(self.M['pre']) < np.mean(self.M['cur']):
                pre_lr = self.G_optimizer.param_groups[0]['lr']
                self.G_optimizer.param_groups[0]['lr'] = max(self.G_optimizer.param_groups[0]['lr'] / 2.0,
                                                             self.lr_lower_boundary)
                self.D_optimizer.param_groups[0]['lr'] = max(self.D_optimizer.param_groups[0]['lr'] / 2.0,
                                                             self.lr_lower_boundary)
                print('M_pre: ' + str(np.mean(self.M['pre'])) + ', M_cur: ' + str(
                    np.mean(self.M['cur'])) + ', lr: ' + str(pre_lr) + ' --> ' + str(
                    self.G_optimizer.param_groups[0]['lr']))
            else:
                print('M_pre: ' + str(np.mean(self.M['pre'])) + ', M_cur: ' + str(np.mean(self.M['cur'])))
                self.M['pre'] = self.M['cur']

                self.M['cur'] = []

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():

                self.generate_image(x_)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")


    def visualize_results(self, epoch, fix=True):
        self.G.eval()



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
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)


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

        print(true_dist.size())
        true_dist = true_dist.cpu().data.numpy()
        samples = self.G(self.sample_z_)
        samples = samples.cpu().data.numpy()

        x = y = np.linspace(-RANGE, RANGE, N_POINTS)
        # plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())

        plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', marker='+')
        plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')
        plt.show()
