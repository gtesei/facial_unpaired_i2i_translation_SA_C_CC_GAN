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
import sys
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
import numpy as np
import os
import random 
from keras.layers import Conv2DTranspose, BatchNormalization
import tensorflow as tf 

from keras.utils import to_categorical

def get_dim_conv(dim,f,p,s):
        return int((dim+2*p-f)/s+1)

def build_generator_enc_dec(img_shape,gf,AU_num,channels,num_layers=4,f_size=6,tranform_layer=False):
    """U-Net Generator"""

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
    G_enc = Model(img,zs)

    #### 
    # = Input(shape=(24, 24, 32))
    #d2_ = Input(shape=(12, 12, 64))
    #d3_ = Input(shape=(6, 6, 128))
    #d4_ = Input(shape=(3, 3, 256))
    
    _zs = [] 
    d_ , c_ = dims.pop()
    #print(0,d_,c_)
    i_ = Input(shape=(d_, d_, c_))
    _zs.append(i_)
    label = Input(shape=(AU_num,), dtype='float32')
    label_r = Reshape((1,1,AU_num))(label)
    
    u = concatenate([i_, label_r],axis=-1)
    
    ## transf 
    if tranform_layer:
        tr = Flatten()(u)
        tr = Dense(c_+AU_num, kernel_initializer='glorot_normal' )(tr)
        tr = LeakyReLU(alpha=0.2)(tr)
        u = Reshape((1,1,c_+AU_num))(tr)
    ##

    u = Conv2D(c_, kernel_size=1, strides=1, padding='valid')(u) ## 1x1 conv 

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


def build_discriminator(img_shape,df,AU_num,num_layers=4,act_multi_label='linear'):

    def d_layer(layer_input, filters, f_size=4, normalization=True,strides=2):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='valid')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = InstanceNormalization()(d)
        return d

    img = Input(shape=img_shape)

    d = img 
    for i in range(num_layers):
        _norm = False if i == 0 else True 
        if i == 0:
            stride = 3
        else:
            stride = 2 
        d = d_layer(d, df*2**i,normalization=_norm,f_size=6,strides=stride)

    flat_repr = Flatten()(d)

    #validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    #print("flat_repr.get_shape().as_list():",flat_repr.get_shape().as_list())
    #print("flat_repr.get_shape().as_list()[1:]:",flat_repr.get_shape().as_list()[1:])

    gan_logit = Dense(df*2**(num_layers-1),kernel_initializer='glorot_normal')(flat_repr)
    gan_logit = LeakyReLU(alpha=0.2)(gan_logit)
    gan_prob = Dense(1, activation='sigmoid')(gan_logit)

    au_logit = Dense(df*2**(num_layers-1),kernel_initializer='glorot_normal')(flat_repr)
    au_logit = LeakyReLU(alpha=0.2)(au_logit)
    au_pred = Dense(AU_num, activation=act_multi_label,kernel_initializer='glorot_normal')(au_logit)

    return Model(img, [gan_prob,au_pred])

if __name__ == '__main__':
    d = build_discriminator(img_shape=(112,112,3),df=64,AU_num=17)
    optimizer = Adam(0.0002, 0.5) 
    print("******** Discriminator/Classifier ********")
    d.summary()
    d.compile(loss=['binary_crossentropy','mse'],
              optimizer=optimizer,
              metrics=['accuracy','mean_squared_error'],
              loss_weights=[1, 1])
    g_enc , g_dec = build_generator_enc_dec(img_shape=(112,112,3),gf=64,
                                            AU_num=17,channels=3,tranform_layer=True)
    print("******** Generator_ENC ********")
    g_enc.summary()
    print("******** Generator_DEC ********")
    g_dec.summary()

