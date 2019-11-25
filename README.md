# Facial Unpaired Image-to-Image Translation with (Self-Attention) Conditional Cycle-Consistent Generative Adversarial Networks

Image-to-image translation is the task of changing a particular aspect of a given image to another.
Typical transformations involve hair, age, gender, skin and facial expressions, where recent advances
in GANs have shown impressive results. Facial unpaired image-to-image translation is the task of
learning to translate an image from a domain (e.g. the face images of a person) captured under
an arbitrary facial expression (e.g. joy) to the same domain but conditioning on a target facial
expression (e.g. surprise), in absence of paired examples. The author already obtained good results performing this task adopting FER2013 as dataset. On the other hand, performing error
analysis, it was noticed that, while there are facial expressions like happy vs. sad mutually exclusive,
there are also fuzzier cases like disgust vs. angry where might not be so clearly exclusive. Even for a
human annotator for many of such cases it might not be so clear and, probably, the correct label should
be both. To address this limitation, in this project we improve the model conditioning on Action
Units (AU) annotations, which describe with continuous vectors the anatomical facial movements
defining a human expression. Specifically, although the number of action units is relatively small (30
AU were found to be anatomically related to the contraction of specific facial muscles), more than
7,000 different AU combinations have been observed. For example, the facial expression for fear is
generally produced with activations of Inner Brow Raiser (AU1), Outer Brow Raiser (AU2), Brow
Lowerer (AU4), Upper Lid Raiser (AU5), Lid Tightener (AU7), Lip Stretcher (AU20) and Jaw Drop
(AU26) (1). As a consequence, the target expression is not described by a categorical variable but
by a continuous vector, leading to deep implications. 

**Note: this is an unpaired image-to-image translation problem.**

## Installation
    $ git https://github.com/gtesei/facial_unpaired_i2i_translation_SA_C_CC_GAN
    $ cd facial_unpaired_i2i_translation_SA_C_CC_GAN/
    $ sudo pip3 install -r requirements.txt

## Train
    $ python train.py
    
    # Defaults
    $ python train.py \
        -d_gan_loss_w 1 \
        -d_cl_loss_w 1 \
        -g_gan_loss_w 2 \
        -g_cl_loss_w 2 \
        -rec_loss_w 1  \
        -adam_lr 0.0002 \
        -adam_beta_1 0.5 \
        -adam_beta_2 0.999 \
        -epochs EPOCHS  170   \
        -batch_size 64   \
        -sample_interval 200 \
        
    # Usage
    $ python train.py -h
    
## Dataset 
We adopt [EmotioNet](https://ieeexplore.ieee.org/abstract/document/7780969), which consists in over 1 million images of facial expressions with associated 
emotion keywords from the Internet and automatically annotated with Action Units (AU). The dataset is re-processed and re-annotated with [OpenFace](https://cmusatyalab.github.io/openface/) to obtain cropped facial images and related AU annotations.

<img src="images/EmotioNet2OpenFace.PNG" align="center" /> 

Here below, the distribution of a sample of 10,753 images processed with OpenFace. 

<img src="images/hist.png" align="center" /> 


## Experiment Log

Id | Code | Description | Notes | 
--- | --- | --- | --- |
e1 | models_1.py, train_1.py | Baseline - Noticed that initialization is a problem. [Epoch 169/170] [Batch 300/10752] [D_gan loss: 7.971193, acc_gan:   0%] [D_AU_loss loss: nan, au_mse: nan] [G_gan loss: 0.000000, G_AU_loss: 00nan, recon: 00nan] time: 0:55:06.274718 -  | adopted [Xavier normal initializization](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).  |
e2 | models_1.py, train_1.py | Anyway, this does not solve the problem. Sometime, it can happen that after some batchs the AU critic loss vanishes. [Epoch 0/1000] [Batch 0/336] [D_gan loss: 0.590275, acc_gan:  56%] [D_AU_loss loss: 2.322111, au_mse: 0.204830] [G_gan loss: 0.000000, G_AU_loss: 00nan, recon: 0.466637] time: 0:00:12.726241 [Epoch 0/1000] [Batch 5/10752] [D_gan loss: 7.971193, acc_gan:   0%] [D_AU_loss loss: nan, au_mse: nan] [G_gan loss: 0.000000, G_AU_loss: 00nan, recon: 00nan] time: 0:00:14.490142  | adopted [Xavier uniform initializization](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf). But again it does not solve the problem every time |
e3 | train_wgan.py , models_wgan.py | __Solved the vanishing loss problem of the critic with Wasserstein GAN or [WGAN](https://arxiv.org/abs/1701.07875)__  | Note that our critic is a multi-task learning function, hence we didn't adopt the convolution at the last layer but we flattened to feed the two separate dense layers. |
e4 | train_gan_custom_au_loss.py | Novel conditional regression lost $log(1-|au1-au2|)$ |  |
e5 | train_gan_pytorch.py , models_gan_pytorch.py  | Re-implemted in PyTorch supporting __loss_nonsaturating__ and __loss_wasserstein_gp__ |  |