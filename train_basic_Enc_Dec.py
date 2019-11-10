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
from keras.utils import to_categorical
import argparse
from sklearn.metrics import accuracy_score
from utils import * 
from keras.layers import Conv2DTranspose, BatchNormalization

###########
def get_dim_conv(dim,f,p,s):
        return int((dim+2*p-f)/s+1)
    
def build_enc_dec(img_shape,gf,AU_num,channels,num_layers=4,f_size=6,tranform_layer=False):

    def conv2d(layer_input, filters, f_size=f_size,strides=2):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='valid')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)
        return d    

    def __deconv2d(layer_input, skip_input, filters, f_size=f_size, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='valid', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u
    
    def deconv2d(layer_input, skip_input, filters, f_size=f_size, dropout_rate=0 , output_padding=None,strides=2):
        """Layers used during upsampling"""
        u = Conv2DTranspose(filters=filters, kernel_size=f_size, 
                            strides=strides, activation='relu' , output_padding=output_padding)(layer_input)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    img = Input(shape=img_shape)

    # Downsampling
    d = img 
    zs = [] 
    dims = []
    _dim = img_shape[0]
    for i in range(num_layers):
        if i == 0:
            stride = 3
        else:
            stride = 2
        d = conv2d(d, gf*2**i,strides=stride)
        zs.append(d)
        _dim = get_dim_conv(_dim,f_size,0,stride)
        dims.append((_dim,gf*2**i))
        #print("D:",_dim,gf*2**i)
    u = Conv2D(AU_num, kernel_size=1, strides=1, padding='valid')(d) ## 1x1 conv 
    u = Reshape((AU_num,))(u)
    zs.append(u)
    G_enc = Model(img,zs)
    
    _zs = [] 
    d_ , c_ = dims.pop()
    i_ = Input(shape=(d_, d_, c_))
    _zs.append(i_)
    label = Input(shape=(AU_num,), dtype='float32')
    #_zs.append(i_)
    u = Reshape((1,1,AU_num))(label)
    #u = concatenate([i_, label_r],axis=-1)
    #_zs.append(u)
    ## transf 
#    if tranform_layer:
#        tr = Flatten()(u)
#        tr = Dense(AU_num, kernel_initializer='glorot_uniform' )(tr)
#        tr = LeakyReLU(alpha=0.2)(tr)
#        u = Reshape((1,1,AU_num))(tr)
    ##

#    u = Conv2D(c_, kernel_size=1, strides=1, padding='valid')(label) ## 1x1 conv 
#    u = Conv2DTranspose(filters=c_, kernel_size=1, 
#                            strides=1, activation='relu')(u)
    u = deconv2d(u, i_, c_,f_size=1,output_padding=None,strides=1)

    # Upsampling
    for i in range(num_layers-1):
        _ch = gf*2**((num_layers-2)-i)
        d_ , c_ = dims.pop()
        #print(i,d_,c_)
        i_ = Input(shape=(d_, d_, c_))
        _zs.append(i_)
        if i == 3:
            u = deconv2d(u, i_, _ch,output_padding=1)
            #u = deconv2d(u, i_, _ch)
        else: 
            u = deconv2d(u, i_, _ch)
        
    #u4 = UpSampling2D(size=2)(u)
    #output_img = Conv2D(channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)
    
    u = Conv2DTranspose(filters=channels, kernel_size=f_size, 
                            strides=3, activation='tanh' , output_padding=1)(u)
     
    
    _zs.reverse()
    _zs.append(label)
    G_dec = Model(_zs,u)

    return G_enc , G_dec

class Enc_Dec():
    def __init__(self, root_data_path, train_size=-1,
        img_rows = 112,img_cols = 112,channels = 3, 
        AU_num=35,
        enc_loss_w=1, dec_loss_w=1, 
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
        self.enc_loss_w = enc_loss_w
        self.dec_loss_w = dec_loss_w

        # optmizer params 
        self.adam_lr = adam_lr
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2

        # Configure data loader
        self.data_loader = InMemoryDataLoader(dataset_name='EmotioNet',
                                                            img_res=self.img_shape,
                                                            root_data_path=self.root_data_path,
                                                            max_images=train_size)
        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        optimizer = Adam(self.adam_lr, self.adam_beta_1, self.adam_beta_2) 

        # Build 
        self.enc , self.dec = build_enc_dec(img_shape=self.img_shape ,gf=64,
                                           AU_num=17,channels=3,tranform_layer=True)
        print("******** Generator_ENC ********")
        self.enc.summary()
        print("******** Generator_DEC ********")
        self.dec.summary()

        # Input images from both domains
        img = Input(shape=self.img_shape)
        label0 = Input(shape=(self.AU_num,))

        # Translate images to the other domain
        z1,z2,z3,z4,a = self.enc(img)
        reconstr = self.dec([z1,z2,z3,z4,label0])
        
        #
        self.combined = Model(inputs=[img,label0],
                              outputs=[ a, reconstr])
        self.combined.compile(loss=['mae','mae'],
                            loss_weights=[  
                            self.enc_loss_w,      # enc loss  
                            self.dec_loss_w      # dec loss 
                            ],
                            optimizer=optimizer)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()
        # logs 
        epoch_history, batch_i_history,  = [] , []   
        enc_loss_history, dec_loss_history = [] , [] 

        for epoch in range(epochs):
            for batch_i, (labels0 , imgs) in enumerate(self.data_loader.load_batch(batch_size=batch_size)):
                
                zs1,zs2,zs3,zs4,a = self.enc.predict(imgs)
                x_hat = self.dec.predict([zs1,zs2,zs3,zs4,labels0])
                
                # 
                comb_loss = self.combined.train_on_batch([imgs, labels0],
                                                        [labels0, imgs])

                elapsed_time = datetime.datetime.now() - start_time

                try:
                    print ("[Epoch %d/%d] [Batch %d/%d] [Enc loss: %f, Dec_loss loss: %f] time: %s " \
                        % ( epoch, epochs,
                            batch_i, self.data_loader.n_batches,
                            comb_loss[0],comb_loss[1],
                            elapsed_time))
                except:
                    print("*** problem to log ***")

                # log
                epoch_history.append(epoch) 
                batch_i_history.append(batch_i)
                enc_loss_history.append(comb_loss[0])
                dec_loss_history.append(comb_loss[1])

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
                    #self.sample_images(epoch, batch_i,use_leo=True)

                    train_history = pd.DataFrame({
                        'epoch': epoch_history, 
                        'batch': batch_i_history, 
                        'enc_loss': enc_loss_history, 
                        'dec_loss' : dec_loss_history
                    })
                    train_history.to_csv(str(sys.argv[0]).split('.')[0]+'_train_log.csv',index=False)

    def sample_images(self, epoch, batch_i):
        for labels0_d , imgs_d in self.data_loader.load_batch(batch_size=1):
            zs1_,zs2_,zs3_,zs4_,a = self.enc.predict(imgs_d)
            
            # Reconstruct image 
            reconstr_ = self.dec.predict([zs1_,zs2_,zs3_,zs4_,labels0_d])
            print("a::",a)
            print("labels0_d::",labels0_d)
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
                    act_au = self.dec.predict([zs1_,zs2_,zs3_,zs4_,au_n])
                    print("labels0_d::",labels0_d)
                    print(r,c,"au_n::",au_n)
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

#if __name__ == '__main__':
#    optimizer = Adam(0.0002, 0.5) 
#    g_enc , g_dec = build_enc_dec(img_shape=(112,112,3),gf=64,
#                                            AU_num=17,channels=3,tranform_layer=True)
#    print("******** Generator_ENC ********")
#    g_enc.summary()
#    print("******** Generator_DEC ********")
#    g_dec.summary()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('-enc_loss_w', help='encoder loss weight', dest='enc_loss_w', type=int, default=1)
    parser.add_argument('-dec_loss_w', help='decoder loss weight', dest='dec_loss_w', type=int, default=1)
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
    gan = Enc_Dec(
        root_data_path = root_data_path,
        train_size = args.train_size, 
        AU_num=17,
        enc_loss_w=args.enc_loss_w, dec_loss_w=args.dec_loss_w, 
        adam_lr=args.adam_lr,adam_beta_1=args.adam_beta_1,adam_beta_2=args.adam_beta_2)
    gan.train(epochs=args.epochs, batch_size=args.batch_size, sample_interval=args.sample_interval)
