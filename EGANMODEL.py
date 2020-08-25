import utilis, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from dataloader import dataloader
import copy

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
            nn.Linear(self.input_dim,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024,128*(self.input_size//4)*(self.input_size//4)),
            nn.BatchNorm1d(128*(self.input_size//4)*(self.input_size//4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,self.output_dim,4,2,1),
            nn.Tanh()
        )
        utilis.initialize_weights(self)


    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1,128,(self.input_size//4),(self.input_size)//4)
        x = self.deconv(x)
        return x

class discriminator(nn.Module):
    def __init__(self, input_dim =1,output_dim=1,input_size = 32):
        super(discriminator,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim,64,4,2,1),
            nn.LeakyReLU(0.2) ,
            nn.Conv2d(64,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size//4)*(self.input_size//4),1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,self.output_dim),
            nn.Sigmoid(),
        )
        utilis.initialize_weights(self)

    def forward(self,input):
        x = self.conv(input)
        x = x.view(-1,128*(self.input_size//4)*(self.input_size//4))
        x = self.fc(x)
        return x


class EGANMODEL(object):
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
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.D = discriminator(input_dim=data.shape[1],output_dim=1, input_size=self.input_size)
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
        self.sample_z_ = torch.rand((self.batch_size,self.z_dim))
        if self.gpu_mode:
            self.sample_z_= self.sample_z_.cuda()

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
                print('The epoch is ',epoch)
                self.G.train()
                epoch_start_time = time.time()
                for iter, (x_, _) in enumerate(self.data_loader):
                    if iter == self.data_loader.dataset.__len__() // self.batch_size:
                        break
                    z_ = torch.rand((self.batch_size, self.z_dim))
                    if self.gpu_mode:
                        x_, z_ = x_.cuda(), z_.cuda()


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
                            self.gen_imgs = self.evalimgs[(i-1)*self.batch_size : i*self.batch_size,:,:,:].detach()

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




                    # # update D network
                    # self.D_optimizer.zero_grad()
                    #
                    # D_real = self.D(x_)
                    # D_real_loss = -torch.mean(D_real)
                    #
                    # G_ = self.G(z_)
                    #
                    # D_fake = self.D(G_)
                    # D_fake_loss = torch.mean(D_fake)
                    #
                    # # gradient penalty
                    # alpha = torch.rand((self.batch_size, 1, 1, 1))
                    # if self.gpu_mode:
                    #     alpha = alpha.cuda()
                    #
                    # x_hat = alpha * x_.data + (1 - alpha) * G_.data
                    # x_hat.requires_grad = True
                    #
                    # pred_hat = self.D(x_hat)
                    # if self.gpu_mode:
                    #     gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                    #                      create_graph=True, retain_graph=True, only_inputs=True)[0]
                    # else:
                    #     gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
                    #                      create_graph=True, retain_graph=True, only_inputs=True)[0]
                    # gradient_penalty = self.lambda_ * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
                    # D_loss = D_real_loss + D_fake_loss + gradient_penalty
                    # D_loss.backward()
                    # self.D_optimizer.step()
                    #
                    # if ((iter + 1) % self.n_critic) == 0:
                    #     # update G network
                    #     self.G_optimizer.zero_grad()
                    #
                    #     G_ = self.G(z_)
                    #     D_fake = self.D(G_)
                    #     G_loss = -torch.mean(D_fake)
                    #     self.train_hist['G_loss'].append(G_loss.item())
                    #
                    #     G_loss.backward()
                    #     self.G_optimizer.step()
                    #
                            self.train_hist['D_loss'].append(D_loss.item())




                    if ((iter + 1) % 100) == 0:
                        print(

                              (epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size)
                self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
                with torch.no_grad():
                    self.visualize_results((epoch + 1))

            self.train_hist['total_time'].append(time.time() - start_time)
            print( 'epoch',
                    np.mean(self.train_hist['per_epoch_time']),
                    self.epoch, self.train_hist['total_time'][0])
            print("Training finish!... save training results")


            self.save()
            utilis.generate_animation(
                        self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                        self.epoch)
            utilis.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name),
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
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
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


        eval_D = eval_fake + eval_real




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
                print('G loss is',criterionG)
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
    # def set_requires_grad(self,nets,requires_grad = False):






