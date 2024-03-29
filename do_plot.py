from __future__ import print_function, division
import scipy

import torch.nn as nn
import torch.nn.functional as F
import torch
import functools

import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import *
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

from models_gan_pytorch_4  import *
from utils import * 

from FID import * 


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
        adam_lr=0.0002,adam_beta_1=0.5,adam_beta_2=0.999,model_name=None):
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
                                                            csv_columns = ['frame', "AU01_c" , "AU02_c"	 , "AU04_c", 
                                                                           "AU05_c", "AU06_c",	 "AU07_c", "AU09_c", 	 
                                                                           "AU10_c",  "AU12_c",  "AU14_c", "AU15_c", 
                                                                           "AU17_c"	,  "AU20_c"	, "AU23_c",	"AU25_c", 
                                                                           "AU26_c" ,  "AU45_c"], 
                                                            max_images=train_size)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> N. images loaded::",len(self.data_loader.lab_vect),"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        #optimizer = Adam(self.adam_lr, self.adam_beta_1, self.adam_beta_2) 
        
        self.a2e = AU2Emotion()

        # Build and compile the discriminators
        self.d = Discriminator(
            64, 'instancenorm', 'lrelu',
            1024, 'none', 'relu', 5, 112
        ).to(device)
        #self.d.init_weights()
        print("******** Discriminator/Classifier ********")
        print(self.d)

        # Build the generators
        self.g = Generator(
            64, 5, 'batchnorm', 'lrelu',
            64, 5, 'batchnorm', 'relu',
            17, 1, 1, 112
        ).to(device)
        #self.g.init_weights()
        print("******** Generator ********")
        print(self.g)
        
        ##
        self.g_optimizer = torch.optim.Adam(self.g.parameters(), self.adam_lr, betas=(self.adam_beta_1, self.adam_beta_2))
        self.d_optimizer = torch.optim.Adam(self.d.parameters(), self.adam_lr, betas=(self.adam_beta_1, self.adam_beta_2))
        ## recover_mode
        self.model_name = model_name
        if self.model_name is not None:
            print(">> recover_mode detected ==> ",model_name)
            adir = os.path.join('saved_models', model_name, 'checkpoint')
            if os.path.exists(adir):
                self.load(os.path.join(adir,'weights.pth'))
            else:
                raise Exception("directory "+str(adir)+" does not exists!")
        else:
            raise Exception("model_name cannot be None!")
            
        
    def train(self, epochs, batch_size=1, sample_interval=50 , save_interval=1000, d_g_ratio=5):

        start_time = datetime.datetime.now()
        # logs 
        epoch_history, batch_i_history,  = [] , []   
        d_gan_loss_history, d_au_loss_history = [], [],
        g_gan_loss_history, g_au_loss_history = [] , [] 
        reconstr_history = [] 
        #
        fid_joy_history, fid_sadness_history, fid_surprise_history, fid_contempt_history = [], [] ,[] ,[] 
        
        ##
        if self.recover_mode:
            print(">> recover_mode detected ==> loading train_history ... ")
            train_history = pd.read_csv(str(sys.argv[0]).split('.')[0]+'_train_log.csv')
            epoch_history = train_history['epoch'].tolist()
            batch_i_history = train_history['batch'].tolist()
            d_gan_loss_history = train_history['d_gan_loss'].tolist()
            d_au_loss_history = train_history['d_AU_loss'].tolist()
            g_gan_loss_history = train_history['g_gan_loss'].tolist()
            g_au_loss_history = train_history['g_AU_loss'].tolist()
            reconstr_history = train_history['reconstr_loss'].tolist()
            fid_joy_history = train_history['fid_joy'].tolist()
            fid_sadness_history = train_history['fid_sadness'].tolist()
            fid_surprise_history = train_history['fid_surprise'].tolist()
            fid_contempt_history = train_history['fid_contempt'].tolist()
            epoch_restart = epoch_history[-1]
            batch_i_restart = batch_i_history[-1]
        else:
            epoch_restart = 0
            batch_i_restart = 0
        ##
        self.g.train()
        self.d.train()

        for epoch in range(epoch_restart,epochs):
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
                elif self.loss_type == 'loss_wasserstein_gp':
                    # train critic 
                    d_loss_dict = train_D_wasserstein_gp(self.g, self.d, imgs, labels0, 
                                           self.lambda_cl, self.lambda_cyc, 
                                           self.data_loader,
                                           device,self.d_optimizer)
                    # train generator 
                    if batch_i % d_g_ratio == 0: 
                        g_loss_dict = train_G_wasserstein_gp(self.g, self.d, imgs, labels0, 
                                           self.lambda_cl, self.lambda_cyc, 
                                           self.data_loader,
                                           device,self.g_optimizer)
                    
                else:
                    raise Exception("Unknown loss type::"+str(self.loss_type))

                torch.cuda.empty_cache() 
                elapsed_time = datetime.datetime.now() - start_time

                try:
                    if batch_i % d_g_ratio == 0:
                        print ("[Epoch %d/%d] [Batch %d/%d] [D_gan loss: %f, D_AU_loss: %f] [G_gan loss: %05f, G_AU_loss: %05f, recon: %05f] time: %s " \
                            % ( epoch, epochs,
                                batch_i, self.data_loader.n_batches(batch_size),
                                d_loss_dict['d_adv_loss'], d_loss_dict['d_cl_loss'],  
                                g_loss_dict['g_adv_loss'],g_loss_dict['g_cl_loss'], g_loss_dict['rec_loss'],  
                                elapsed_time))
                    else:
                        print ("[Epoch %d/%d] [Batch %d/%d] [D_gan loss: %f, D_AU_loss: %f] time: %s " \
                            % ( epoch, epochs,
                                batch_i, self.data_loader.n_batches(batch_size),
                                d_loss_dict['d_adv_loss'], d_loss_dict['d_cl_loss'],  
                                elapsed_time))
                except:
                    print("*** problem to log ***")

                # log & save generated image samples
                if batch_i % sample_interval == 0:
                    with torch.no_grad():
                        self.g.eval()
                        self.sample_images(epoch, batch_i)
                        ##
                        try: 
                            fis_dict = self.measure_fis(epoch,sample_size=1000)
                        except Exception as e:
                            print("Exception occurred::",e)
                            print("Trying again ...")
                            fis_dict = self.measure_fis(epoch,sample_size=1000)
                        fid_joy_history.append(fis_dict['fid_joy'])
                        fid_sadness_history.append(fis_dict['fid_sadness'])
                        fid_surprise_history.append(fis_dict['fid_surprise'])
                        fid_contempt_history.append(fis_dict['fid_contempt'])
                        ##
                        epoch_history.append(epoch)
                        batch_i_history.append(batch_i)
                        d_gan_loss_history.append(d_loss_dict['d_adv_loss'].cpu().detach().numpy())
                        d_au_loss_history.append(d_loss_dict['d_cl_loss'].cpu().detach().numpy())
                        g_gan_loss_history.append(g_loss_dict['g_adv_loss'].cpu().detach().numpy())
                        g_au_loss_history.append(g_loss_dict['g_cl_loss'].cpu().detach().numpy())
                        reconstr_history.append(g_loss_dict['rec_loss'].cpu().detach().numpy())
                        ##
                        self.g.train()

                    train_history = pd.DataFrame({
                        'epoch': epoch_history, 
                        'batch': batch_i_history, 
                        'd_gan_loss': d_gan_loss_history, 
                        'd_AU_loss': d_au_loss_history, 
                        'g_gan_loss': g_gan_loss_history, 
                        'g_AU_loss': g_au_loss_history, 
                        'reconstr_loss': reconstr_history, 
                        'fid_joy': fid_joy_history, 
                        'fid_sadness': fid_sadness_history, 
                        'fid_surprise': fid_surprise_history, 
                        'fid_contempt': fid_contempt_history
                    })
                    train_history.to_csv(str(sys.argv[0]).split('.')[0]+'_train_log.csv',index=False)
                # save 
                if batch_i % sample_interval == 0:
                    adir = os.path.join('saved_models', str(sys.argv[0]).split('.')[0], 'checkpoint')
                    if not os.path.exists(adir):
                        os.makedirs(adir)
                    #self.save(os.path.join(adir,'weights.{:d}.pth'.format(epoch)))
                    self.save(os.path.join(adir,'weights.pth'))
    
    def measure_fis(self, epoch,sample_size=1000,emotions = ["joy", "sadness", "surprise", "contempt"]):
        fis_dict = {}
        for batch_i, (labels0 , imgs) in enumerate(self.data_loader.load_batch(batch_size=sample_size)):
                imgs = np.transpose(imgs,(0,3,1,2))
                dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor 
                labels0, imgs = torch.tensor(labels0).to(device).type(dtype), torch.tensor(imgs).to(device).type(dtype)
                zs = self.g.encode(imgs)
                for em in emotions:
                    print("****",em,"****")
                    idx = self.a2e.get_idx(self.data_loader.lab_vect,emotion=em)
                    images = self.data_loader.img_vect[idx.squeeze()]
                    images = images[0:sample_size]
                    #
                    au_em = self.a2e.emotion2aus(em,sample_size)
                    au_em = torch.tensor(au_em).to(device).type(dtype)
                    emo_img = self.g.decode(zs,au_em)
                    emo_img = torch.clamp(emo_img, min=0, max=1000)
                    emo_img = emo_img.cpu().detach().numpy()
                    emo_img = np.transpose(emo_img,(0,2,3,1))
                    print("images",images.shape)
                    print("emo_img",emo_img.shape)
                    fid_value = calculate_fid(images, emo_img, False, 8)
                    print("fid_value",fid_value,type(fid_value))
                    #
                    fis_dict['fid_'+em] = fid_value
                    torch.cuda.empty_cache()
                break 
        return fis_dict
    
    def save(self, path):
        states = {
            'G': self.g.state_dict(),
            'D': self.d.state_dict(),
            'optim_G': self.g_optimizer.state_dict(),
            'optim_D': self.d_optimizer.state_dict()
        }
        torch.save(states, path)
        
    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.g.load_state_dict(states['G'])
        if 'D' in states:
            self.d.load_state_dict(states['D'])
        if 'optim_G' in states:
            self.g_optimizer.load_state_dict(states['optim_G'])
        if 'optim_D' in states:
            self.d_optimizer.load_state_dict(states['optim_D'])
    
    def saveG(self, path):
        states = {
            'G': self.g.state_dict()
        }
        torch.save(states, path)

    def sample_images(self, n_samples):
        n_samp = 1 
        for labels0 , imgs in self.data_loader.load_batch(batch_size=1):
            ## disc
            imgs_d = np.transpose(imgs,(0,3,1,2))
            dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor 
            labels0_d, imgs_d = torch.tensor(labels0).to(device).type(dtype), torch.tensor(imgs_d).to(device).type(dtype)
            #gan_pred_prob,au_prob = self.d(imgs_d)
            #des_au_1 = torch.tensor(self.data_loader.gen_rand_cond(batch_size=1)).to(device).type(dtype)
            des_au_1 = torch.tensor(self.data_loader.gen_rand_cond_for_binary_au(labels0)).to(device).type(dtype)[0]
            
            # Translate images 
            zs = self.g.encode(imgs_d)
            
            # Reconstruct image
            #print("labels0_d",labels0_d.shape)
            reconstr_ = self.g.decode(zs,labels0_d)

            # Transl. image 
            transl_ = self.g.decode(zs,des_au_1)
            
            ## save reconstraction 
            if not os.path.exists('plot_images'):
                os.makedirs('plot_images')
            #plot reconstr_   
            reconstr_ = reconstr_.cpu().detach().numpy()
            reconstr_ = np.transpose(reconstr_,(0,2,3,1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plot_grid(np.concatenate([imgs, reconstr_]), 
                          row_titles=None, 
                          col_titles=["Orig.[n_samp:%d]" % (n_samp),'Reconstr.'],
                          nrow = 1,ncol = 2,
                          save_filename="plot_images/reconstr_%d_%d.png" % (n_samp, 1))

            #plot transl_   
            transl_ = transl_.cpu().detach().numpy()
            transl_ = np.transpose(transl_,(0,2,3,1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plot_grid(np.concatenate([imgs, transl_]), 
                          row_titles=None, 
                          col_titles=["Orig.[n_samp:%d]" % (n_samp),'Transl.'],
                          nrow = 1,ncol = 2,
                          save_filename="plot_images/translat_%d_%d.png" % (n_samp, 2))
            #### AU 
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
                    act_au = self.g.decode(zs,au_n)
                    act_au = act_au.cpu().detach().numpy()
                    act_au = np.transpose(act_au,(0,2,3,1))
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
                          save_filename="plot_images/au_edition_%d_%d.png" % (n_samp, 3))
            
            #### joy, sadness, surprise, contempt
            n_row = 1 #  
            n_col = 5 # 
            emotions = ["joy", "sadness", "surprise", "contempt"]
            em_images = np.repeat(imgs,n_row*n_col,axis=0)
            n = 0 
            for r in range(n_row):
                for c in range(n_col):
                    if n > 0: 
                        au_em = self.a2e.emotion2aus(emotions[n-1],1)
                        au_em = torch.tensor(au_em).to(device).type(dtype)
                        #
                        #print("au_em",au_em.shape)
                        emo_img = self.g.decode(zs,au_em)
                        emo_img = emo_img.cpu().detach().numpy()
                        emo_img = np.transpose(emo_img,(0,2,3,1))
                        em_images[n,:] = emo_img
                    n += 1 
            col_names = ["Orig.", "Joy", "Sadness", "Surprise", "Contempt"]
            plot_grid(em_images,
                  #row_titles=[0,.33],
                  col_titles=col_names,
                  nrow = 1,ncol = 5,save_filename="plot_images/emotion_trans_%d_%d.png" % (n_samp, 4))
            n_samp += 1 
            if n_samp >= n_samples:
                break 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('-lambda_cl', help='loss weight for cond. regress. loss', dest='lambda_cl', type=float, default=10)
    parser.add_argument('-lambda_cyc', help='reconstr. loss weight', dest='lambda_cyc', type=float, default=10)
    parser.add_argument('-loss_type', help='loss type [loss_nonsaturating] ', dest='loss_type', type=str, default='loss_wasserstein_gp')
    parser.add_argument('-d_g_ratio', help='# train iterations of critic per each train iteration of generator', dest='d_g_ratio', type=int, default=1)
    parser.add_argument('-adam_lr', help='Adam l.r.', dest='adam_lr', type=float, default=0.0002)
    parser.add_argument('-adam_beta_1', help='Adam beta-1', dest='adam_beta_1', type=float, default=0.5)
    parser.add_argument('-adam_beta_2', help='Adam beta-2', dest='adam_beta_2', type=float, default=0.999)
    parser.add_argument('-root_data_path', help='base file path', dest='root_data_path', type=str, default='datasets')
    parser.add_argument('-model_name', help='base file path', dest='model_name', type=str, default='train_gan_pytorch_4')
    parser.add_argument('-n_samples', help='number of samples', dest='n_samples', type=int, default=100)
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
        train_size = -1, 
        AU_num=17,
        lambda_cl=args.lambda_cl,lambda_cyc=args.lambda_cyc,
        loss_type=args.loss_type,
        adam_lr=args.adam_lr,adam_beta_1=args.adam_beta_1,adam_beta_2=args.adam_beta_2,model_name=args.model_name)
    gan.sample_images(n_samples=args.n_samples)
    
