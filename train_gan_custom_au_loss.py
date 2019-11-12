from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import Reshape
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import InMemoryDataLoader
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
import numpy as np
import pandas as pd 
import os
import random 

import warnings
import tensorflow as tf 
import keras.backend as K
from keras.utils import to_categorical
import argparse
from sklearn.metrics import accuracy_score

from models_gan import *
from utils import * 



def log_mean_absolute_error(y_true, y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return K.mean(  K.log(1. - K.abs(y_pred - y_true)  ) , axis=-1)



class C_CC_GAN():
    def __init__(self, root_data_path, train_size=-1,
        img_rows = 112,img_cols = 112,channels = 3, 
        AU_num=35,
        d_gan_loss_w=1,d_cl_loss_w=1,
        g_gan_loss_w=1,g_cl_loss_w=1,
        rec_loss_w=1,
        adam_lr=0.0002,adam_beta_1=0.5,adam_beta_2=0.999):
        # paths 
        self.root_data_path = root_data_path 
        # Input shape
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.AU_num = AU_num

        # Loss weights 
        self.d_gan_loss_w = d_gan_loss_w
        self.d_cl_loss_w = d_cl_loss_w
        self.g_gan_loss_w = g_gan_loss_w
        self.g_cl_loss_w = g_cl_loss_w
        self.rec_loss_w = rec_loss_w

        # optmizer params 
        self.adam_lr = adam_lr
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2

        # Configure data loader
        self.data_loader = InMemoryDataLoader(dataset_name='EmotioNet',
                                                            img_res=self.img_shape,
                                                            root_data_path=self.root_data_path,
                                                            normalize=True,
                                                            max_images=train_size)
        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        optimizer = Adam(self.adam_lr, self.adam_beta_1, self.adam_beta_2) 

        # Build and compile the discriminators
        self.d = build_discriminator(img_shape=self.img_shape,df=64,AU_num=self.AU_num,act_multi_label='sigmoid')
        print("******** Discriminator/Classifier ********")
        self.d.summary()
        self.d.compile(loss=[
            'binary_crossentropy',  # gan
             log_mean_absolute_error   # AU regression  
           ],
            optimizer=optimizer,
            metrics=['accuracy','accuracy'],
            loss_weights=[
            self.d_gan_loss_w , # gan
            self.d_cl_loss_w   # AU regression  
            ])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_enc , self.g_dec = build_generator_enc_dec(img_shape=self.img_shape,gf=64,AU_num=self.AU_num,channels=self.channels,
                                                          tranform_layer=True)
        print("******** Generator_ENC ********")
        self.g_enc.summary()
        print("******** Generator_DEC ********")
        self.g_dec.summary()

        # Input images from both domains
        img = Input(shape=self.img_shape)
        label0 = Input(shape=(self.AU_num,))
        label1 = Input(shape=(self.AU_num,))

        # Translate images to the other domain
        z1,z2,z3,z4 = self.g_enc(img)
        fake = self.g_dec([z1,z2,z3,z4,label1])

        # Translate images back to original domain
        reconstr = self.g_dec([z1,z2,z3,z4,label0])

        # For the combined model we will only train the generators
        self.d.trainable = False

        # Discriminators determines validity of translated images gan_prob,class_prob [label,img], [gan_prob,class_prob]
        gan_valid , AU_valid = self.d(fake)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img,label0,label1],
                              outputs=[ gan_valid, AU_valid, 
                                        reconstr])
        self.combined.compile(loss=['binary_crossentropy', log_mean_absolute_error,
                                    'mae'],
                            loss_weights=[  
                            self.g_gan_loss_w ,                 # g_loss gan 
                            self.g_cl_loss_w  ,                 # au loss  
                            self.rec_loss_w                     # reconstruction loss
                            ],
                            optimizer=optimizer)

    def train(self, epochs, batch_size=1, sample_interval=50 , d_g_ratio=5):

        start_time = datetime.datetime.now()
        # logs 
        epoch_history, batch_i_history,  = [] , []   
        d_gan_loss_history, d_gan_accuracy_history, d_au_loss_history, d_au_mse_history = [], [], [], [] 
        g_gan_loss_history, g_au_loss_history = [] , [] 
        reconstr_history = [] 

        # Adversarial loss ground truths
        valid = np.ones((batch_size,1) )
        fake = np.zeros((batch_size,1) )

        for epoch in range(epochs):
            for batch_i, (labels0 , imgs) in enumerate(self.data_loader.load_batch(batch_size=batch_size)):
                des_au_1 = self.data_loader.gen_rand_cond(batch_size=batch_size)
                des_au_2 = self.data_loader.gen_rand_cond(batch_size=batch_size)
                des_au_3 = self.data_loader.gen_rand_cond(batch_size=batch_size)
                des_au_4 = self.data_loader.gen_rand_cond(batch_size=batch_size)
                des_au_5 = self.data_loader.gen_rand_cond(batch_size=batch_size)
                #
                des_au_6 = self.data_loader.gen_rand_cond(batch_size=batch_size)
                des_au_7 = self.data_loader.gen_rand_cond(batch_size=batch_size)
                des_au_8 = self.data_loader.gen_rand_cond(batch_size=batch_size) 
                des_au_9 = self.data_loader.gen_rand_cond(batch_size=batch_size)
                des_au_10 = self.data_loader.gen_rand_cond(batch_size=batch_size)
                
                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                zs1,zs2,zs3,zs4 = self.g_enc.predict(imgs)
                fakes_1 = self.g_dec.predict([zs1,zs2,zs3,zs4,des_au_1])
                fakes_2 = self.g_dec.predict([zs1,zs2,zs3,zs4,des_au_2])
                fakes_3 = self.g_dec.predict([zs1,zs2,zs3,zs4,des_au_3])
                fakes_4 = self.g_dec.predict([zs1,zs2,zs3,zs4,des_au_4])
                fakes_5 = self.g_dec.predict([zs1,zs2,zs3,zs4,des_au_5])
                #
                fakes_6 = self.g_dec.predict([zs1,zs2,zs3,zs4,des_au_6])
                fakes_7 = self.g_dec.predict([zs1,zs2,zs3,zs4,des_au_7])
                fakes_8 = self.g_dec.predict([zs1,zs2,zs3,zs4,des_au_8])
                fakes_9 = self.g_dec.predict([zs1,zs2,zs3,zs4,des_au_9])
                fakes_10 = self.g_dec.predict([zs1,zs2,zs3,zs4,des_au_10])

            
                # Train the discriminators (original images = real / translated = Fake)
                idx = np.random.permutation(11*labels0.shape[0])
                all_au = np.concatenate([labels0,des_au_1,des_au_2,des_au_3,des_au_4,des_au_5,des_au_6,des_au_7,des_au_8,des_au_9,des_au_10])
                all_imgs = np.concatenate([imgs,fakes_1,fakes_2,fakes_3,fakes_4,fakes_5,fakes_6,fakes_7,fakes_8,fakes_9,fakes_10])
                gan_labels = np.concatenate([valid,fake,fake,fake,fake,fake,fake,fake,fake,fake,fake])
                # shuffle 
                all_au = all_au[idx]
                all_imgs = all_imgs[idx]
                gan_labels = gan_labels[idx]

                d_loss  = self.d.train_on_batch(all_imgs, [gan_labels,all_au])

                if batch_i % d_g_ratio == 0:

                    # ------------------
                    #  Train Generators
                    # ------------------
                    _imgs = np.concatenate([
                        imgs, imgs, imgs, imgs, imgs, imgs, imgs, imgs, imgs, imgs])

                    _labels0_cat = np.concatenate([labels0, labels0, labels0, labels0, labels0, labels0, labels0, labels0, labels0, labels0])

                    _labels1_all_other = np.concatenate([des_au_1,des_au_2,des_au_3,des_au_4,des_au_5,des_au_6,des_au_7,des_au_8,des_au_9,des_au_10])

                    # I know this should be outside the loop; left here to make code more understandable 
                    _valid = np.concatenate([valid,valid,valid,valid,valid,valid,valid,valid,valid,valid])

                    idx = np.random.permutation(_imgs.shape[0])
                    _imgs = _imgs[idx]
                    _labels0_cat = _labels0_cat[idx]
                    _labels1_all_other = _labels1_all_other[idx]
                    _valid = _valid[idx]

                    # Train the generators
                    g_loss = self.combined.train_on_batch([_imgs, _labels0_cat, _labels1_all_other],
                                                            [_valid, _labels1_all_other, _imgs])

                    elapsed_time = datetime.datetime.now() - start_time

                    try:
                        print ("[Epoch %d/%d] [Batch %d/%d] [D_gan loss: %f, acc_gan: %3d%%] [D_AU_loss loss: %f, au_mse: %f] [G_gan loss: %05f, G_AU_loss: %05f, recon: %05f] time: %s " \
                            % ( epoch, epochs,
                                batch_i, self.data_loader.n_batches,
                                d_loss[1],100*d_loss[3],d_loss[2],d_loss[4],
                                g_loss[1],g_loss[2],g_loss[3],
                                elapsed_time))
                    except:
                        print("*** problem to log ***")

                    # log
                    epoch_history.append(epoch) 
                    batch_i_history.append(batch_i)
                    d_gan_loss_history.append(d_loss[1])
                    d_gan_accuracy_history.append(100*d_loss[3])
                    d_au_loss_history.append(d_loss[2])
                    d_au_mse_history.append(100*d_loss[4])
                    g_gan_loss_history.append(g_loss[1])
                    g_au_loss_history.append(g_loss[2])
                    reconstr_history.append(g_loss[3])

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
                    #self.sample_images(epoch, batch_i,use_leo=True)

                    train_history = pd.DataFrame({
                        'epoch': epoch_history, 
                        'batch': batch_i_history, 
                        'd_gan_loss': d_gan_loss_history, 
                        'd_gan_accuracy' : d_gan_accuracy_history,
                        'd_AU_loss': d_au_loss_history, 
                        'd_AU_MSE': d_au_mse_history, 
                        'g_gan_loss': g_gan_loss_history, 
                        'g_AU_loss': g_au_loss_history, 
                        'reconstr_loss': reconstr_history
                    })
                    train_history.to_csv(str(sys.argv[0]).split('.')[0]+'_train_log.csv',index=False)

    def sample_images(self, epoch, batch_i):
        for labels0_d , imgs_d in self.data_loader.load_batch(batch_size=1):
            ## disc
            gan_pred_prob,au_prob = self.d.predict(imgs_d)
            
            # Translate images 
            zs1_,zs2_,zs3_,zs4_ = self.g_enc.predict(imgs_d)
            
            # Reconstruct image 
            reconstr_ = self.g_dec.predict([zs1_,zs2_,zs3_,zs4_,labels0_d])
            
            ## save reconstraction 
            if not os.path.exists('log_images'):
                os.makedirs('log_images')
            #plot    
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plot_grid(np.concatenate([imgs_d, reconstr_]), 
                          row_titles=None, 
                          col_titles=["Orig.[ep:%d]" % (epoch),'Reconstr.'],
                          nrow = 1,ncol = 2,
                          save_filename="log_images/reconstr_%d_%d.png" % (epoch, batch_i))
            ####
            n_row = 4 # alpha 
            n_col = 9 # AUs 
            col_names = ['AU1_r','AU2_r','AU4_r','AU5_r','AU10_r',
                         'AU12_r','AU15_r','AU25_r','AU45_r']
            col_idx = [0,1,2,3,7,8,10,14,16] 
            assert len(col_names) == len(col_idx)
            alphas = [0,.33,.66,1]
            au_grid = np.repeat(labels0_d,n_row*n_col,axis=0)
            img_tens = np.repeat(imgs_d,n_row*n_col,axis=0)
            n = 0 
            for r in range(n_row):
                for c in range(n_col):
                    au_n = au_grid[[n],:]
                    au_n[0,col_idx[c]] = alphas[r]
                    #
                    act_au = self.g_dec.predict([zs1_,zs2_,zs3_,zs4_,au_n])
                    img_tens[n,:] = act_au
                    n += 1 
            #plot    
            col_names_plot = ['AU1','AU2','AU4','AU5','AU10',
                         'AU12','AU15','AU25','AU45']
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
    parser.add_argument('-d_gan_loss_w', help='loss weight for discrim. real/fake', dest='d_gan_loss_w', type=int, default=1)
    parser.add_argument('-d_cl_loss_w', help='loss weight for discrim. multiclass', dest='d_cl_loss_w', type=int, default=1)
    parser.add_argument('-g_gan_loss_w', help='loss weight for gen. real/fake', dest='g_gan_loss_w', type=int, default=2)
    parser.add_argument('-g_cl_loss_w', help='loss weight for gen. multiclass', dest='g_cl_loss_w', type=int, default=2)
    parser.add_argument('-rec_loss_w', help='reconstr. loss weight', dest='rec_loss_w', type=int, default=1)
    parser.add_argument('-adam_lr', help='Adam l.r.', dest='adam_lr', type=float, default=0.0002)
    parser.add_argument('-adam_beta_1', help='Adam beta-1', dest='adam_beta_1', type=float, default=0.5)
    parser.add_argument('-adam_beta_2', help='Adam beta-2', dest='adam_beta_2', type=float, default=0.999)
    parser.add_argument('-epochs', help='N. epochs', dest='epochs', type=int, default=170)
    parser.add_argument('-batch_size', help='batch size', dest='batch_size', type=int, default=64)
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
        d_gan_loss_w=args.d_gan_loss_w,d_cl_loss_w=args.d_cl_loss_w,
        g_gan_loss_w=args.g_gan_loss_w,g_cl_loss_w=args.g_cl_loss_w,
        rec_loss_w=args.rec_loss_w,
        adam_lr=args.adam_lr,adam_beta_1=args.adam_beta_1,adam_beta_2=args.adam_beta_2)
    gan.train(epochs=args.epochs, batch_size=args.batch_size, sample_interval=args.sample_interval)
