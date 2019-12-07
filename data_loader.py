import scipy
from glob import glob
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os 
import cv2
from utils import * 
from os import listdir 
from os.path import isfile, join, isdir     


class AU2Emotion():
    #https://imotions.com/blog/facial-action-coding-system/
    def __init__(self):
        self.aus =        ["AU01_c" , "AU02_c"	 , "AU04_c", 
                           "AU05_c", "AU06_c",	 "AU07_c", "AU09_c", 	 
                           "AU10_c",  "AU12_c",  "AU14_c", "AU15_c", 
                           "AU17_c"	,  "AU20_c"	, "AU23_c",	"AU25_c", 
                           "AU26_c" ,  "AU45_c"]
        self.au2idx = {1:0, 2:1, 4:2, 
                       5:3, 6:4, 7:5, 9:6, 
                       10:7, 12:8, 14:9, 15:10, 
                       17:11, 20:12, 23:13, 25:14, 
                       26:15, 45:17}
        self.emotion2au = {"joy": [6,12],
                           "sadness": [1, 4, 15],
                           "surprise": [1, 2, 5, 26],
                           "fear": [1,2,4,5,7,20,26],
                           "anger": [4, 5, 7, 23],
                           #"disgust": [9,15,16],
                           "contempt": [12,14]}
    
    def emotion2aus(self,emotion,batch_size):
        au_list = self.emotion2au[emotion]
        au_list_idx = [self.au2idx[i] for i in au_list]
        au_vect = np.zeros((batch_size,len(self.aus)))
        for i in range(len(self.aus)):
            if i in au_list_idx:
                au_vect[:,i] = 1 
        return au_vect
    def get_idx(self,lab_vect,emotion):
        au_list = self.emotion2au[emotion]
        au_list_idx = [self.au2idx[i] for i in au_list]
        idx = np.argwhere(np.take(lab_vect,au_list_idx,axis=1).sum(axis=1)==len(au_list_idx))
        return idx 
    
    
def get_00000_num(num):
    if num < 10:
        return '00000'+str(num)
    elif num < 100:
        return '0000'+str(num)
    elif num < 1000:
        return '000'+str(num)
    elif num < 10000:
        return '00'+str(num)
    elif num < 100000:
        return '0'+str(num)
    elif num < 1000000:
        return str(num)
    else:
        raise Exception("number too high:"+str(num))

class InMemoryDataLoader():
    def __init__(self, 
                 dataset_name, 
                 img_res=(112, 112,3),
                 root_data_path=None,
#                 path_image_dir=None, 
                 normalize=True,
#                 csv_columns = ['frame',  'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 
#                                   'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r', 
#                                   'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 
#                                   'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c'],
                 csv_columns = ['frame',  'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 
                                   'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'],
                 max_images=-1,
                 image_patter_fn = 'frame_det_00_FRAME_ID.bmp'):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.root_data_path = root_data_path
        #self.path_csv = path_csv 
        #self.path_image_dir = path_image_dir 
        self.csv_columns = csv_columns
        self.max_images = max_images
        self.image_patter_fn = image_patter_fn # image_patter_fn.replace('FRAME_ID','1')
        ## 
        self.normalize = normalize 
        ## load dataset 
        self._load_internally()
        
    def gen_rand_cond(self,batch_size=1,add_noise=False):
        idx = np.random.choice(self.lab_vect.shape[0],size=batch_size)
        cond = self.lab_vect[idx]
        if add_noise: 
            cond += np.random.uniform(-0.1, 0.1, cond.shape)
        cond = np.clip(a=cond,a_min=0,a_max=5)
        return cond
    
    def gen_rand_cond_for_binary_au(self,bt):
        au_num = bt.shape[1]
        alist = [] 
        for i in range(au_num):
            _bt = bt.copy()
            _bt[:,i] = np.ones_like(_bt[:,i]) - _bt[:,i]
            alist.append(_bt)
        #cond = np.concatenate(alist,axis=0)
        return alist
    
    def _process_data_dir(self, 
                          im_dir, 
                          other_dir='processed', 
                          csv_fn='EmotioNet.csv', 
                          img_dirn='EmotioNet_aligned'):
        labels = pd.read_csv(join(self.root_data_path,im_dir,other_dir,csv_fn))
        labels.columns = [i.strip() for i in labels.columns]
        print(">> removing",np.sum(labels['success']==0),"images [success==0] ...")
        labels = labels[labels['success']==1]
        labels = labels[self.csv_columns]
        labels.reset_index(inplace=True,drop=True)
        frame_list = labels.iloc[:,0].tolist()
        lab_vect = labels.iloc[:,1:].to_numpy()
        assert len(frame_list) == len(lab_vect)
        #
        n_images = min(len(frame_list),self.max_images) if self.max_images > 0 else len(frame_list)
        print(">loading",n_images,"images ...")
        lab_vect = lab_vect[:n_images]
        img_vect = np.zeros((n_images,
                                 self.img_res[0],
                                 self.img_res[1],
                                 self.img_res[2]) , 'float32')
        for i in range(n_images):
            img_path = os.path.join(*[self.root_data_path,im_dir,other_dir,img_dirn,
                                      self.image_patter_fn.replace('FRAME_ID',get_00000_num(frame_list[i]))])
            img = read_cv2_img(img_path)
            if self.normalize:
                #img = img/127.5 - 1.
                img = img/255 - 0.
            img_vect[i] = img
            if i % 100 == 0:
                print(i,end=' ... ')
        #
        assert np.sum(np.isnan(lab_vect)) == 0 
        assert np.sum(np.isnan(img_vect)) == 0
        #
        return lab_vect, img_vect
                
    def _load_internally(self):
        print(">> loading "+str(self.dataset_name)+" ...") 
        if self.dataset_name == 'EmotioNet':
            lab_vect_list , img_vect_list = [] , [] 
            im_dirs = [d for d in listdir(self.root_data_path) if isdir(join(self.root_data_path,d))]
            print(">>> found",len(im_dirs),"directories::",im_dirs)
            for k in range(len(im_dirs)):
                im_dir = im_dirs[k]
                if im_dir != '.git':
                    print(k,"===============>>",im_dir)
                    lab_vect, img_vect = self._process_data_dir(im_dir)
                    lab_vect_list.append(lab_vect)
                    img_vect_list.append(img_vect)
                
            ##
            self.lab_vect = np.concatenate(lab_vect_list,axis=0)
            #self.lab_vect = self.lab_vect / self.lab_vect.max()
            self.lab_vect = np.where(np.max(self.lab_vect, axis=0)==0, self.lab_vect, self.lab_vect*1./np.max(self.lab_vect, axis=0))
            self.img_vect = np.concatenate(img_vect_list,axis=0)
            print("lab_vect::",lab_vect.shape,"  -- img_vect::",img_vect.shape)
        else:
            raise Exception("dataset not supported:"+str(self.dataset_name))
        
    def n_batches(self,batch_size):
        return int(len(self.img_vect) / batch_size)
    
    def load_batch(self, batch_size=1, flip_prob=0, is_testing=False):
        if is_testing:
            raise Exception("not supported yet!")
        n_batches = self.n_batches(batch_size)
        for i in range(n_batches):
            idx = np.random.choice(self.lab_vect.shape[0],size=batch_size)
            #print("idx",idx)
            batch_images = self.img_vect[idx]
            labels = self.lab_vect[idx]
            if flip_prob > 0:
                for i in range(batch_size):
                    if np.random.random() > 0.5:
                        batch_images[i] = np.fliplr(batch_images[i])
            yield labels , batch_images
    


if __name__ == '__main__':
    root_data_path = 'datasets/'
    #csv_filename = 'EmotioNet.csv'
    #images_dir = 'EmotioNet_aligned' 
    base_path = os.path.abspath(os.path.dirname(root_data_path))
    #csv_path = os.path.join(*[base_path,csv_filename])
    #img_path = os.path.join(*[base_path,images_dir])
    ## 
    dl = InMemoryDataLoader(dataset_name='EmotioNet',
                            img_res=(112, 112,3),
                            #path_csv=csv_path,
                            #path_image_dir=img_path, 
                            root_data_path=root_data_path, 
#                            csv_columns = ['frame', "AU01_c" , "AU02_c"	 , "AU04_c", 
#                                                                           "AU05_c", "AU06_c",	 "AU07_c", "AU09_c", 	 
#                                                                           "AU10_c",  "AU12_c",  "AU14_c", "AU15_c", 
#                                                                           "AU17_c"	,  "AU20_c"	, "AU23_c",	"AU25_c", 
#                                                                           "AU26_c" ,  "AU45_c"], 
                            max_images=12)
    ## 
    print(dl.gen_rand_cond(batch_size=2).shape)
    for batch_i, (labels , batch_images) in enumerate(dl.load_batch(batch_size=4)):
        img_lab = ["batch:"+str(batch_i)+"_"+str(ii) for ii in range(4)]
        show_images_row(batch_images, img_lab, rows=1)
    for batch_i, (labels , batch_images) in enumerate(dl.load_batch(batch_size=8)):
        img_lab = ["batch:"+str(batch_i)+"_"+str(ii) for ii in range(8)]
        show_images_row(batch_images, img_lab, rows=2,save_filename='data_loader_test')
        plot_au(batch_images[0], labels[0], title='Data loader test')
        plot_grid(batch_images,
                  row_titles=[0,.33],
                  col_titles=['AU1','AU2','AU3','AU4'],
                  nrow = 2,ncol = 4,save_filename='data_loader_test_2')
    for batch_i, (labels , batch_images) in enumerate(dl.load_batch(batch_size=3)):
        al = dl.gen_rand_cond_for_binary_au(labels)
        print(al)
        print("len",len(al),al[0].shape)
        al = dl.gen_rand_cond(3)
        print("***********",al)
        break 
    ## 
    a2e = AU2Emotion()
    for batch_i, (labels , batch_images) in enumerate(dl.load_batch(batch_size=8)):
        _labels = ["orig."]+list(a2e.emotion2au.keys())
        plot_grid(batch_images,
                  #row_titles=[0,.33],
                  col_titles=_labels,
                  nrow = 1,ncol = 7,save_filename=None)
        break 
    joy = a2e.emotion2aus('joy',1).reshape(17)
    dl.lab_vect[dl.lab_vect==a2e.emotion2aus('joy',1)]
    idx = a2e.get_idx(dl.lab_vect,emotion='joy')
    joy_img = dl.lab_vect[idx]
    plot_grid(dl.img_vect[idx.squeeze(),:,:,:],
                  #row_titles=[0,.33],
                  col_titles=None,
                  nrow = 1,ncol = 7,save_filename=None)
    
    
    
    
        
         
