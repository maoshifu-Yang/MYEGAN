import utilis, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from dataloader import dataloader
import copy
import matplotlib.pyplot as plt
import random

import torchvision
#%%


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
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128, 2),
        )

        utilis.initialize_weights(self)


    def forward(self, input):
        x = self.fc(input)
        return x

class discriminator(nn.Module):
    def __init__(self, input_dim =1,output_dim=1,input_size = 32):
        super(discriminator,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim),
            nn.Sigmoid(),
        )
        utilis.initialize_weights(self)

    def forward(self,input):

        x = self.fc(input)
        return x


class EGAN_TOY(object):
    def __init__(self,args):
        #parameter
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
        self.datasize = 500



        #先加上 loss的形式
        self.loss_mode = ['orignal','LS','minimax']
        self.candinum = 2


        self.z_dim = 62
        self.lambda_ = 10
        self.n_critic = 2 #the number of iterations of the critic per generator iteration


        #load dataset
        self.data_loader= dataloader(self.dataset,self.input_size, self.batch_size)
        data = self.data_loader.__iter__().__next__()[0]

        #network init
        self.G = generator(input_dim=self.z_dim, output_dim=2, input_size=self.input_size)
        self.D = discriminator(input_dim=2,output_dim=1, input_size=self.input_size)
        self.G_optimizer = optim.Adam(self.G.parameters(),lr=args.lrG,betas=(args.beta1,args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(),lr=args.lrD,betas=(args.beta1,args.beta2))

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
        utilis.print_network(self.G)
        utilis.print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise
        self.sample_z_ = torch.randn((self.datasize,self.z_dim))
        if self.gpu_mode:
            self.sample_z_= self.sample_z_.cuda()

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
                # print('The epoch is ',epoch)
                self.G.train()
                epoch_start_time = time.time()

                data = self.toy_dataset('25gaussians',size=256)
                x_ = torch.from_numpy(data).cuda()
                z_ = torch.randn(self.datasize,self.z_dim).cuda()






                for i in range(2):
                        #update G
                        if i == 0:
                            self.Fitness,self.evalimgs,self.sel_mut = self.Evo_G(x_,z_)
                            self.evalimgs = torch.cat(self.evalimgs, dim=0)
                            shuffle_ids = torch.randperm(self.evalimgs.size()[0])

                            self.evalimgs = self.evalimgs[shuffle_ids]
                            # print(self.evalimgs.size(),'evalimgs size ')

                        else:
                            self.D_optimizer.zero_grad()
                            self.gen_imgs = self.evalimgs[(i-1)*self.datasize : i*self.datasize,:].detach()

                            # print(self.evalimgs.size(),'evalimgs in else size')
                            # print(self.gen_imgs.size(),'gen imgs size')


                            self.real_out = self.D(x_)
                            D_real_loss = self.BCE(self.real_out,self.y_real_)
                            self.fake_out = self.D(self.gen_imgs)
                            # print('D_fake is',D_fake.size())
                            D_fake_loss = self.BCE(self.fake_out,self.y_fake_)
                            D_loss = D_real_loss+D_fake_loss

                            D_loss.backward()
                            self.D_optimizer.step()







                with torch.no_grad():
                    if epoch%100==0:
                        self.generate_image((data))



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

        #load current best G
        F = self.Fitness[:,2]
        idx = np.where(F==max(F))[0][0]
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
        utilis.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')



    def fitness_score(self,eval_fake_imgs,eval_real_imgs):

        eval_fake = self.D(eval_fake_imgs)
        eval_real = self.D(eval_real_imgs)

        fake_loss = self.BCE(eval_fake,self.y_fake_)
        real_loss = self.BCE(eval_real,self.y_real_)

        D_loss_score = fake_loss + real_loss

        #quality fitness score
        Fq = nn.functional.sigmoid(eval_fake).data.mean().cpu().numpy()


        #Diversity fitness score






        gradients = torch.autograd.grad(outputs = D_loss_score,inputs = self.D.parameters(),grad_outputs=torch.ones(D_loss_score.size()).to(self.device),
                                        create_graph=True,retain_graph=True,only_inputs=True)



        with torch.no_grad():
            for i,grad in enumerate(gradients):
                grad = grad.view(-1)
                allgrad = grad if i==0 else torch.cat([allgrad,grad])

        Fd = -torch.log(torch.norm(allgrad)).data.cpu().numpy()

        return Fq,Fd


    def G_loss(self,Dfake,loss_type):

        if loss_type == 'LS':
            gloss = self.MSE_loss(Dfake,self.y_real_)

        if loss_type == 'minimax':

            gloss = self.BCE(Dfake,self.y_fake_)

        if loss_type == 'orignal':


            gloss = self.BCE(Dfake,self.y_real_)

        return gloss


    def Evo_G(self,inputimage,z_):

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




                g_loss = self.G_loss(self.fake_out,criterionG)
                g_loss.backward()
                self.G_optimizer.step()

                with torch.no_grad():
                    eval_fake_imgs = self.G(z_)



                Fq,Fd = self.fitness_score(eval_fake_imgs,inputimage)

                F = Fq + lamba_f * Fd


                if count<self.candinum:
                    F_list[count] = F #关于F的列表
                    Fit_list.append([Fq,Fd,F])  #把当前的放进去
                    G_list.append(copy.deepcopy(self.G.state_dict()))# 生成器优化器的参数
                    optG_list.append(copy.deepcopy(self.G_optimizer.state_dict())) #生成器网络参数
                    eval_fake_image_list.append(eval_fake_imgs)
                    selectmutation.append(self.loss_mode[j])
                #update
                # candi_num 是生存下来的数量
                else:  #下一代就选到里面
                    fit_com = F-F_list
                    if max(fit_com)>0:
                        ids_replace = np.where(fit_com==max(fit_com))[0][0]
                        F_list[ids_replace] = F   #
                        Fit_list[ids_replace] = [Fq,Fd,F]
                        G_list[ids_replace] = copy.deepcopy(self.G.state_dict())
                        optG_list[ids_replace] = copy.deepcopy(self.G_optimizer.state_dict())
                        eval_fake_image_list[ids_replace] = eval_fake_imgs
                        selectmutation[ids_replace] = self.loss_mode[j]

                count = count + 1
        self.candis = copy.deepcopy(G_list)
        self.opt_candis = copy.deepcopy(optG_list)


        return np.array(Fit_list),eval_fake_image_list,selectmutation
    #

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