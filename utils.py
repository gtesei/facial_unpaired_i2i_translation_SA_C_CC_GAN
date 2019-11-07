import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import cv2

import matplotlib.gridspec as gridspec



def read_cv2_img(path):
    '''
    Read color images
    :param path: Path to image
    :return: Only returns color images
    '''
    img = cv2.imread(path, -1)

    if img is not None:
        if len(img.shape) != 3:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    # im = cv2.resize(im, (48, 48))

    return img

def show_cv2_img(img, title='img'):
    '''
    Display cv2 image
    :param img: cv::mat
    :param title: title
    :return: None
    '''
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_images_row(imgs, titles, rows=1,save_filename=None):
    '''
       Display grid of cv2 images image
       :param img: list [cv::mat]
       :param title: titles
       :return: None
    '''
    assert ((titles is None) or (len(imgs) == len(titles)))
    num_images = len(imgs)

    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, num_images + 1)]

    fig = plt.figure()
    for n, (image, title) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(rows, np.ceil(num_images / float(rows)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        ax.set_title(title)
        plt.axis('off')
    if save_filename is None:
        plt.show()
    else:
        fig.savefig("%s.png" % (save_filename))
        
def plot_au(img, aus, title=None):
    '''
    Plot action units
    :param img: HxWx3
    :param aus: N
    :return:
    '''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    fig.subplots_adjust(0, 0, 0.8, 1)  # get rid of margins

    # display img
    ax.imshow(img)

    if len(aus) == 11:
        au_ids = ['1','2','4','5','6','9','12','17','20','25','26']
        x = 0.1
        y = 0.39
        i = 0
        for au, id in zip(aus, au_ids):
            if id == '9':
                x = 0.5
                y -= .15
                i = 0
            elif id == '12':
                x = 0.1
                y -= .15
                i = 0

            ax.text(x + i * 0.2, y, id, horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='r', fontsize=20)
            ax.text((x-0.001)+i*0.2, y-0.07, au, horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='b', fontsize=20)
            i+=1

    else:
        au_ids = ['1', '2', '4', '5', '6', '7', '9', '10', '12', '14', '15', '17', '20', '23', '25', '26', '45']
        x = 0.1
        y = 0.39
        i = 0
        for au, id in zip(aus, au_ids):
            if id == '9' or id == '20':
                x = 0.1
                y -= .15
                i = 0

            ax.text(x + i * 0.2, y, id, horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='r', fontsize=20)
            ax.text((x-0.001)+i*0.2, y-0.07, au, horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='b', fontsize=20)
            i+=1

    if title is not None:
        ax.text(0.5, 0.95, title, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='r', fontsize=20)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #fig.savefig("%s.png" % ('tt'))
    plt.close(fig)

    return data

def plot_grid(imgs,title=None,row_titles=None,col_titles=None,nrow = 10,ncol = 20,save_filename=None):
    fig = plt.figure(figsize=(ncol+1, nrow+1)) 
    gs = gridspec.GridSpec(nrow, ncol,
             wspace=0.03, hspace=0.03, 
             top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
             left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 
    n = 0
    for i in range(nrow):
        for j in range(ncol):
            im = imgs[n]
            ax= plt.subplot(gs[i,j],aspect='equal')
            if i == 0 and col_titles is not None:
                ax.set_title(col_titles[j])
            if j == 0 and row_titles is not None:
                ax.set_ylabel(row_titles[i])
            ax.imshow(im)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            #plt.axis('off')
            n += 1
    if title is not None:
        plt.title(title)
    if save_filename is None:
        plt.show()
    else:
        fig.savefig("%s.png" % (save_filename))

    
    
        
         
