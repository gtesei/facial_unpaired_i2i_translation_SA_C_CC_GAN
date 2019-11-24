from __future__ import print_function, division
import scipy

import torch.nn as nn
import torch.nn.functional as F
import torch
import functools

import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import InMemoryDataLoader
import numpy as np
import pandas as pd 
import os
import random 
import argparse
import os
import time
import torch
import torchvision
import tqdm

import warnings

import argparse
from sklearn.metrics import accuracy_score

from models_gan_pytorch import *
from utils import * 


# reproducibility
torch.manual_seed(777)
np.random.seed(777)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class C_CC_GAN():
    def __init__(self, root_data_path, train_size=-1,
        img_rows = 112,img_cols = 112,channels = 3, 
        AU_num=35,
        lambda_cl=1,lambda_cyc=1,
        loss_type='loss_nonsaturating',
        adam_lr=0.0002,adam_beta_1=0.5,adam_beta_2=0.999):
        # paths 
        self.root_data_path = root_data_path 
        # Input shape
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.img_shape = (self.channels,self.img_rows, self.img_cols)
        self.AU_num = AU_num

        # Loss weights 
        self.lambda_cl = lambda_cl
        self.lambda_cyc = lambda_cyc
        
        # loss type 
        self.loss_type = loss_type

        # optmizer params 
        self.adam_lr = adam_lr
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2

        # Configure data loader
        self.data_loader = InMemoryDataLoader(dataset_name='EmotioNet',
                                                            img_res=(self.img_rows, self.img_cols,self.channels), 
                                                            root_data_path=self.root_data_path,
                                                            normalize=True,
                                                            max_images=train_size)

        #optimizer = Adam(self.adam_lr, self.adam_beta_1, self.adam_beta_2) 

        # Build and compile the discriminators
        self.d = Discriminator(img_shape=self.img_shape,df=64,AU_num=self.AU_num).to(device)
        self.d.init_weights()
        print("******** Discriminator/Classifier ********")
        print(self.d)

        # Build the generators
        self.g = Generator(img_shape=(3,112,112),gf=64,AU_num=self.AU_num).to(device)
        self.g.init_weights()
        print("******** Generator ********")
        print(self.g)
        
        ##
        self.g_optimizer = torch.optim.Adam(self.g.parameters(), self.adam_lr, betas=(self.adam_beta_1, self.adam_beta_2))
        self.d_optimizer = torch.optim.Adam(self.d.parameters(), self.adam_lr, betas=(self.adam_beta_1, self.adam_beta_2))


    def train(self, epochs, batch_size=1, sample_interval=50 , d_g_ratio=5):

        start_time = datetime.datetime.now()
        # logs 
        epoch_history, batch_i_history,  = [] , []   
        d_gan_loss_history, d_au_loss_history = [], [],
        g_gan_loss_history, g_au_loss_history = [] , [] 
        reconstr_history = [] 

        ##
        self.g.train()
        self.d.train()

        for epoch in range(epochs):
            for batch_i, (labels0 , imgs) in enumerate(self.data_loader.load_batch(batch_size=batch_size)):
                imgs = np.transpose(imgs,(0,3,1,2))
                dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor 
                labels0, imgs = torch.tensor(labels0).to(device).type(dtype), torch.tensor(imgs).to(device).type(dtype)
                if self.loss_type == 'loss_nonsaturating':
                    d_loss , d_loss_dict , g_loss, g_loss_dict = loss_nonsaturating(self.g, self.d, 
                                                                                    imgs, labels0, 
                                                                                    self.lambda_cl, self.lambda_cyc, 
                                                                                    self.data_loader,
                                                                                    device,
                                                                                    train_generator=(batch_i % d_g_ratio == 0))
                    ## opt. discr. 
                    self.d_optimizer.zero_grad()
                    d_loss.backward(retain_graph=True)
                    self.d_optimizer.step()
                    ## opt. gen.
                    if g_loss is not None:
                        self.g_optimizer.zero_grad()
                        g_loss.backward()
                        self.g_optimizer.step()
                else:
                    raise Exception("Unknown loss type::"+str(self.loss_type))

                torch.cuda.empty_cache() 
                elapsed_time = datetime.datetime.now() - start_time

                try:
                    if g_loss is not None:
                        print ("[Epoch %d/%d] [Batch %d/%d] [D_gan loss: %f, D_AU_loss: %f] [G_gan loss: %05f, G_AU_loss: %05f, recon: %05f] time: %s " \
                            % ( epoch, epochs,
                                batch_i, self.data_loader.n_batches,
                                d_loss_dict['d_adv_loss'], d_loss_dict['d_cl_loss'],  
                                g_loss_dict['g_adv_loss'],g_loss_dict['g_cl_loss'], g_loss_dict['rec_loss'],  
                                elapsed_time))
                    else:
                        print ("[Epoch %d/%d] [Batch %d/%d] [D_gan loss: %f, D_AU_loss: %f] time: %s " \
                            % ( epoch, epochs,
                                batch_i, self.data_loader.n_batches,
                                d_loss_dict['d_adv_loss'], d_loss_dict['d_cl_loss'],  
                                elapsed_time))
                except:
                    print("*** problem to log ***")

                # log
                if g_loss is not None:
                    epoch_history.append(epoch) 
                    batch_i_history.append(batch_i)
                    d_gan_loss_history.append(d_loss_dict['d_adv_loss'].cpu().detach().numpy())
                    d_au_loss_history.append(d_loss_dict['d_cl_loss'].cpu().detach().numpy())
                    g_gan_loss_history.append(g_loss_dict['g_adv_loss'].cpu().detach().numpy())
                    g_au_loss_history.append(g_loss_dict['g_cl_loss'].cpu().detach().numpy())
                    reconstr_history.append(g_loss_dict['rec_loss'].cpu().detach().numpy())

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    with torch.no_grad():
                        self.g.eval()
                        self.sample_images(epoch, batch_i)
                        #self.sample_images(epoch, batch_i,use_leo=True)
                        self.g.train()

                    train_history = pd.DataFrame({
                        'epoch': epoch_history, 
                        'batch': batch_i_history, 
                        'd_gan_loss': d_gan_loss_history, 
                        'd_AU_loss': d_au_loss_history, 
                        'g_gan_loss': g_gan_loss_history, 
                        'g_AU_loss': g_au_loss_history, 
                        'reconstr_loss': reconstr_history
                    })
                    train_history.to_csv(str(sys.argv[0]).split('.')[0]+'_train_log.csv',index=False)

    def sample_images(self, epoch, batch_i):
        for labels0 , imgs in self.data_loader.load_batch(batch_size=1):
            ## disc
            imgs_d = np.transpose(imgs,(0,3,1,2))
            dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor 
            labels0_d, imgs_d = torch.tensor(labels0).to(device).type(dtype), torch.tensor(imgs_d).to(device).type(dtype)
            #gan_pred_prob,au_prob = self.d(imgs_d)
            des_au_1 = torch.tensor(self.data_loader.gen_rand_cond(batch_size=1)).to(device).type(dtype)
            
            # Translate images 
            zs = self.g.encode(imgs_d)
            
            # Reconstruct image 
            reconstr_ = self.g.translate_decode(zs,labels0_d)

            # Transl. image 
            transl_ = self.g.translate_decode(zs,des_au_1)
            
            ## save reconstraction 
            if not os.path.exists('log_images'):
                os.makedirs('log_images')
            #plot reconstr_   
            reconstr_ = reconstr_.cpu()
            reconstr_ = np.transpose(reconstr_.detach().numpy(),(0,2,3,1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plot_grid(np.concatenate([imgs, reconstr_]), 
                          row_titles=None, 
                          col_titles=["Orig.[ep:%d]" % (epoch),'Reconstr.'],
                          nrow = 1,ncol = 2,
                          save_filename="log_images/reconstr_%d_%d.png" % (epoch, batch_i))

            #plot transl_   
            transl_ = transl_.cpu()
            transl_ = np.transpose(transl_.detach().numpy(),(0,2,3,1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plot_grid(np.concatenate([imgs, transl_]), 
                          row_titles=None, 
                          col_titles=["Orig.[ep:%d]" % (epoch),'Transl.'],
                          nrow = 1,ncol = 2,
                          save_filename="log_images/translat_%d_%d.png" % (epoch, batch_i))
            ####
            n_row = 4 # alpha 
            n_col = 9 # AUs 
            col_names = ['AU1_r','AU2_r','AU4_r','AU5_r','AU10_r',
                         'AU12_r','AU15_r','AU25_r','AU45_r']
            col_idx = [0,1,2,3,7,8,10,14,16] 
            assert len(col_names) == len(col_idx)
            alphas = [0,.33,.66,1]
            au_grid = np.repeat(labels0,n_row*n_col,axis=0)
            img_tens = np.repeat(imgs,n_row*n_col,axis=0)
            n = 0 
            for r in range(n_row):
                for c in range(n_col):
                    au_n = au_grid[[n],:]
                    au_n[0,col_idx[c]] = alphas[r]
                    au_n = torch.tensor(au_n).to(device).type(dtype)
                    #
                    act_au = self.g.translate_decode(zs,au_n)
                    act_au = act_au.cpu()
                    act_au = np.transpose(act_au.detach().numpy(),(0,2,3,1))
                    act_au = act_au
                    img_tens[n,:] = act_au
                    n += 1 
            #plot    
            col_names_plot = ['AU1','AU2','AU4','AU5','AU10','AU12','AU15','AU25','AU45']
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plot_grid(img_tens, 
                          row_titles=alphas, 
                          col_titles=col_names_plot,
                          nrow = n_row,ncol = n_col,
                          save_filename="log_images/au_edition_%d_%d.png" % (epoch, batch_i))
            break 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('-lambda_cl', help='loss weight for cond. regress. loss', dest='lambda_cl', type=float, default=1.)
    parser.add_argument('-lambda_cyc', help='reconstr. loss weight', dest='lambda_cyc', type=float, default=1.)
    parser.add_argument('-loss_type', help='loss type [loss_nonsaturating] ', dest='loss_type', type=str, default='loss_nonsaturating')
    parser.add_argument('-adam_lr', help='Adam l.r.', dest='adam_lr', type=float, default=0.0002)
    parser.add_argument('-adam_beta_1', help='Adam beta-1', dest='adam_beta_1', type=float, default=0.5)
    parser.add_argument('-adam_beta_2', help='Adam beta-2', dest='adam_beta_2', type=float, default=0.999)
    parser.add_argument('-epochs', help='N. epochs', dest='epochs', type=int, default=170)
    parser.add_argument('-batch_size', help='batch size', dest='batch_size', type=int, default=32)
    parser.add_argument('-sample_interval', help='sample interval', dest='sample_interval', type=int, default=1000)
    parser.add_argument('-root_data_path', help='base file path', dest='root_data_path', type=str, default='datasets')
    parser.add_argument('-train_size', help='train size [-1 for all train data]', dest='train_size', type=int, default=-1)
    args = parser.parse_args()
    
    # print parameters
    print('-' * 30)
    print('Parameters .')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    # GAN 
    root_data_path = args.root_data_path
    gan = C_CC_GAN(
        root_data_path = root_data_path,
        train_size = args.train_size, 
        AU_num=17,
        lambda_cl=args.lambda_cl,lambda_cyc=args.lambda_cyc,
        loss_type=args.loss_type,
        adam_lr=args.adam_lr,adam_beta_1=args.adam_beta_1,adam_beta_2=args.adam_beta_2)
    gan.train(epochs=args.epochs, batch_size=args.batch_size, sample_interval=args.sample_interval)
