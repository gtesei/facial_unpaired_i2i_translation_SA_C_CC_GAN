# Facial Unpaired Image-to-Image Translation with (Self-Attention) Conditional Cycle-Consistent Generative Adversarial Networks

Image-to-image translation is the task of changing a particular aspect of a given image to another.
Typical transformations involve hair, age, gender, skin and facial expressions, where recent advances
in GANs have shown impressive results. Facial unpaired image-to-image translation is the task of
learning to translate an image from a domain (e.g. the face images of a person) captured under
an arbitrary facial expression (e.g. joy) to the same domain but conditioning on a target facial
expression (e.g. surprise), in absence of paired examples. The author already obtained good results
(25) performing this task adopting FER2013 (9) as dataset. On the other hand, performing error
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
by a continuous vector, leading to deep implications. In section 1.1.3 the differences between this
project, author’s CS230 project and other most related works are discussed.

**Note: this is an unpaired image-to-image translation problem.**

## Installation
    $ git https://github.com/gtesei/facial_unpaired_i2i_translation_SA_C_CC_GAN
    $ cd facial_unpaired_i2i_translation_SA_C_CC_GAN/
    $ sudo pip3 install -r requirements.txt

## Train
    $ python ccyclegan_t26.py
    
    # Defaults
    $ python ccyclegan_t26.py \
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
    $ python ccyclegan_t26.py -h
    
