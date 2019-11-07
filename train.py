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

import tensorflow as tf 

from keras.utils import to_categorical
import argparse
from sklearn.metrics import accuracy_score

from  models import *

class C_CC_GAN():
    def __init__(self, base_path, csv_path, img_path,
        img_rows = 112,img_cols = 112,channels = 3, 
        AU_num=35,
        d_gan_loss_w=1,d_cl_loss_w=1,
        g_gan_loss_w=1,g_cl_loss_w=1,
        rec_loss_w=1,
        adam_lr=0.0002,adam_beta_1=0.5,adam_beta_2=0.999):
        # paths 
        self.base_path = base_path
        self.csv_path = csv_path
        self.img_path = img_path
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
                                                            path_csv=self.csv_path,
                                                            path_image_dir=self.img_path, 
                                                            max_images=12)
        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        optimizer = Adam(self.adam_lr, self.adam_beta_1, self.adam_beta_2) 

        # Build and compile the discriminators
        self.d = build_discriminator(img_shape=self.img_shape,df=64,AU_num=self.AU_num,act_multi_label='linear')
        print("******** Discriminator/Classifier ********")
        self.d.summary()
        self.d.compile(loss=[
            'binary_crossentropy',  # gan
            'mse'   # AU regression  
           ],
            optimizer=optimizer,
            metrics=['accuracy','mean_squared_error'],
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
        self.combined.compile(loss=['binary_crossentropy','mse',
                                    'mae'],
                            loss_weights=[  
                            self.g_gan_loss_w ,                 # g_loss gan 
                            self.g_cl_loss_w  ,                 # g_loss class  
                            self.rec_loss_w                     # reconstruction loss
                            ],
                            optimizer=optimizer)
    
    def generate_new_labels(self,labels0):
        labels1 = [] 
        for i in range(len(labels0)):
            allowed_values = list(range(0, self.AU_num))
            allowed_values.remove(labels0[i])
            labels1.append(random.choice(allowed_values))
        return np.array(labels1,'int32')
    
    def generate_new_labels_all(self,labels0):
        labels_all = [] 
        for i in range(len(labels0)):
            allowed_values = list(range(0, self.AU_num))
            allowed_values.remove(labels0[i])
            labels_all.append(np.array(allowed_values,'int32'))
        return np.array(labels_all,'int32')

    def train(self, epochs, batch_size=1, sample_interval=50 , d_g_ratio=5):

        start_time = datetime.datetime.now()
        # logs 
        epoch_history, batch_i_history,  = [] , []   
        d_gan_loss_history, d_gan_accuracy_history, d_cl_loss_history, d_cl_accuracy_history = [], [], [], [] 
        g_gan_loss_history, g_cl_loss_history = [] , [] 
        reconstr_history = [] 

        # Adversarial loss ground truths
        valid = np.ones((batch_size,1) )
        fake = np.zeros((batch_size,1) )

        null_labels = np.zeros((batch_size,7) )

        for epoch in range(epochs):
            for batch_i, (labels0 , imgs) in enumerate(self.data_loader.load_batch(batch_size=batch_size)):
                labels1_all = self.generate_new_labels_all(labels0)

                labels0_cat = to_categorical(labels0, AU_num=self.AU_num)
                #
                labels1_all_1 = to_categorical(labels1_all[:,0], AU_num=self.AU_num)
                labels1_all_2 = to_categorical(labels1_all[:,1], AU_num=self.AU_num)
                labels1_all_3 = to_categorical(labels1_all[:,2], AU_num=self.AU_num)
                labels1_all_4 = to_categorical(labels1_all[:,3], AU_num=self.AU_num)
                labels1_all_5 = to_categorical(labels1_all[:,4], AU_num=self.AU_num)
                labels1_all_6 = to_categorical(labels1_all[:,5], AU_num=self.AU_num)
                
                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                zs1,zs2,zs3,zs4 = self.g_enc.predict(imgs)
                fakes_1 = self.g_dec.predict([zs1,zs2,zs3,zs4,labels1_all_1])
                fakes_2 = self.g_dec.predict([zs1,zs2,zs3,zs4,labels1_all_2])
                fakes_3 = self.g_dec.predict([zs1,zs2,zs3,zs4,labels1_all_3])
                fakes_4 = self.g_dec.predict([zs1,zs2,zs3,zs4,labels1_all_4])
                fakes_5 = self.g_dec.predict([zs1,zs2,zs3,zs4,labels1_all_5])
                fakes_6 = self.g_dec.predict([zs1,zs2,zs3,zs4,labels1_all_6])

                # Train the discriminators (original images = real / translated = Fake)
                idx = np.random.permutation(self.num_classes*labels0.shape[0])
                _labels_cat = np.concatenate([labels0_cat,
                                              null_labels,
                                              null_labels,
                                              null_labels,
                                              null_labels,
                                              null_labels,
                                              null_labels])
                _imgs = np.concatenate([imgs,
                                        fakes_1,
                                        fakes_2,
                                        fakes_3,
                                        fakes_4,
                                        fakes_5,
                                        fakes_6])
                _vf = np.concatenate([valid,fake,fake,fake,fake,fake,fake])
                _labels_cat = _labels_cat[idx]
                _imgs = _imgs[idx]
                _vf = _vf[idx]

                d_loss  = self.d.train_on_batch(_imgs, [_vf,_labels_cat])

                if batch_i % d_g_ratio == 0:

                    # ------------------
                    #  Train Generators
                    # ------------------
                    _imgs = np.concatenate([
                        imgs,
                        imgs,
                        imgs,
                        imgs,
                        imgs,
                        imgs])

                    _labels0_cat = np.concatenate([
                        labels0_cat,
                        labels0_cat,
                        labels0_cat,
                        labels0_cat,
                        labels0_cat,
                        labels0_cat])

                    _labels1_all_other = np.concatenate([
                        labels1_all_1, 
                        labels1_all_2,
                        labels1_all_3,
                        labels1_all_4,
                        labels1_all_5,
                        labels1_all_6])

                    # I know this should be outside the loop; left here to make code more understandable 
                    _valid = np.concatenate([
                        valid,
                        valid,
                        valid,
                        valid,
                        valid,
                        valid])

                    idx = np.random.permutation((self.num_classes-1)*labels0.shape[0])
                    _imgs = _imgs[idx]
                    _labels0_cat = _labels0_cat[idx]
                    _labels1_all_other = _labels1_all_other[idx]
                    _valid = _valid[idx]

                    # Train the generators
                    g_loss = self.combined.train_on_batch([_imgs, _labels0_cat, _labels1_all_other],
                                                            [_valid, _labels1_all_other, _imgs])

                    elapsed_time = datetime.datetime.now() - start_time

                    print ("[Epoch %d/%d] [Batch %d/%d] [D_gan loss: %f, acc_gan: %3d%%] [D_cl loss: %f, acc_cl: %3d%%] [G_gan loss: %05f, G_cl: %05f, recon: %05f] time: %s " \
                        % ( epoch, epochs,
                            batch_i, self.data_loader.n_batches,
                            d_loss[1],100*d_loss[3],d_loss[2],100*d_loss[4],
                            g_loss[1],g_loss[2],g_loss[3],
                            elapsed_time))

                    # log
                    epoch_history.append(epoch) 
                    batch_i_history.append(batch_i)
                    d_gan_loss_history.append(d_loss[1])
                    d_gan_accuracy_history.append(100*d_loss[3])
                    d_cl_loss_history.append(d_loss[2])
                    d_cl_accuracy_history.append(100*d_loss[4])
                    g_gan_loss_history.append(g_loss[1])
                    g_cl_loss_history.append(g_loss[2])
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
                        'd_cl_loss': d_cl_loss_history, 
                        'd_cl_accuracy': d_cl_accuracy_history, 
                        'g_gan_loss': g_gan_loss_history, 
                        'g_cl_loss': g_cl_loss_history, 
                        'reconstr_loss': reconstr_history
                    })
                    train_history.to_csv(str(sys.argv[0]).split('.')[0]+'_train_log.csv',index=False)

    def sample_images(self, epoch, batch_i, use_leo=False):
        ## disc
        labels0_d , imgs_d = self.data_loader.load_data(batch_size=64, is_testing=True)

        gan_pred_prob,class_pred_prob = self.d.predict(imgs_d)
        
        gan_pred = (gan_pred_prob > 0.5)*1.0
        gan_pred = gan_pred.reshape((64,))
        
        class_pred = np.argmax(class_pred_prob,axis=1)

        gan_test_accuracy = accuracy_score(y_true=np.ones(64), y_pred=gan_pred)
        class_test_accuracy = accuracy_score(y_true=labels0_d, y_pred=class_pred)

        print("*** TEST *** [D_gan accuracy   :",gan_test_accuracy,"] [D_cl accuracy :",class_test_accuracy,"]")

        ## gen         
        if use_leo:
            labels0_ , imgs_ = self.data_loader.load_leo()
        else:
            labels0_ , imgs_ = self.data_loader.load_data(batch_size=1, is_testing=True)
        labels1_all = self.generate_new_labels_all(labels0_)

        labels0_cat = to_categorical(labels0_, num_classes=self.num_classes)
        labels1_all_1 = to_categorical(labels1_all[:,0], num_classes=self.num_classes)
        labels1_all_2 = to_categorical(labels1_all[:,1], num_classes=self.num_classes)
        labels1_all_3 = to_categorical(labels1_all[:,2], num_classes=self.num_classes)
        labels1_all_4 = to_categorical(labels1_all[:,3], num_classes=self.num_classes)
        labels1_all_5 = to_categorical(labels1_all[:,4], num_classes=self.num_classes)
        labels1_all_6 = to_categorical(labels1_all[:,5], num_classes=self.num_classes)

        # Translate images 
        zs1_,zs2_,zs3_,zs4_ = self.g_enc.predict(imgs_)
        fake_1 = self.g_dec.predict([zs1_,zs2_,zs3_,zs4_,labels1_all_1])
        fake_2 = self.g_dec.predict([zs1_,zs2_,zs3_,zs4_,labels1_all_2])
        fake_3 = self.g_dec.predict([zs1_,zs2_,zs3_,zs4_,labels1_all_3])
        fake_4 = self.g_dec.predict([zs1_,zs2_,zs3_,zs4_,labels1_all_4])
        fake_5 = self.g_dec.predict([zs1_,zs2_,zs3_,zs4_,labels1_all_5])
        fake_6 = self.g_dec.predict([zs1_,zs2_,zs3_,zs4_,labels1_all_6])
      
        # Reconstruct image 
        reconstr_ = self.g_dec.predict([zs1_,zs2_,zs3_,zs4_,labels0_cat])

        gen_imgs = np.concatenate([imgs_, 
                                   fake_1, 
                                   fake_2, 
                                   fake_3, 
                                   fake_4, 
                                   fake_5, 
                                   fake_6,
                                   reconstr_])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Orig:'+str(self.lab_dict[labels0_.item(0)]), 
                  'Trans:'+str(self.lab_dict[labels1_all[:,0].item(0)]),
                  'Trans:'+str(self.lab_dict[labels1_all[:,1].item(0)]),
                  'Trans:'+str(self.lab_dict[labels1_all[:,2].item(0)]),
                  'Trans:'+str(self.lab_dict[labels1_all[:,3].item(0)]),
                  'Trans:'+str(self.lab_dict[labels1_all[:,4].item(0)]),
                  'Trans:'+str(self.lab_dict[labels1_all[:,5].item(0)]),
                  'Reconstr.']
        r, c = 2, 4
        fig, axs = plt.subplots(r, c)
        
        plt.subplots_adjust(hspace=0)

        if not os.path.exists( "images/%s/"% (self.dataset_name)):
            os.makedirs( "images/%s/"% (self.dataset_name)  )
        
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt].reshape((self.img_rows,self.img_cols)),cmap='gray')
                axs[i,j].set_title(titles[cnt])
                axs[i,j].axis('off')
                cnt += 1

        if use_leo:
            fig.savefig("images/%s/%d_%d_leo.png" % (self.dataset_name, epoch, batch_i))
        else:
            fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CCycleGAN')
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
    parser.add_argument('-sample_interval', help='sample interval', dest='sample_interval', type=int, default=200)
    parser.add_argument('-file_path', help='base file path', dest='file_path', type=str, default='datasets/sample/')
    parser.add_argument('-csv_filename', help='csv filename', dest='csv_filename', type=str, default='csv_filename')
    parser.add_argument('-images_dir', help='images directory', dest='images_dir', type=str, default='images_aligned')
    args = parser.parse_args()
    
    # print parameters
    print('-' * 30)
    print('Parameters .')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    # GAN 
    base_path = os.path.abspath(os.path.dirname(args.file_path))
    csv_path = os.path.join(*[base_path,args.csv_filename])
    img_path = os.path.join(*[base_path,args.images_dir])
    gan = C_CC_GAN(
        base_path = base_path,
        csv_path = csv_path, 
        img_path = img_path, 
        d_gan_loss_w=args.d_gan_loss_w,d_cl_loss_w=args.d_cl_loss_w,
        g_gan_loss_w=args.g_gan_loss_w,g_cl_loss_w=args.g_cl_loss_w,
        rec_loss_w=args.rec_loss_w,
        adam_lr=args.adam_lr,adam_beta_1=args.adam_beta_1,adam_beta_2=args.adam_beta_2)
    gan.train(epochs=args.epochs, batch_size=args.batch_size, sample_interval=args.sample_interval)
