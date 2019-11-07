import scipy
from glob import glob
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os 
import cv2
from utils import * 
    
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
                 path_csv=None,
                 path_image_dir=None, 
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
        self.path_csv = path_csv 
        self.path_image_dir = path_image_dir 
        self.csv_columns = csv_columns
        self.max_images = max_images
        self.image_patter_fn = image_patter_fn # image_patter_fn.replace('FRAME_ID','1')
        ## 
        self.normalize = normalize 
        ## load dataset 
        self._load_internally()
        
    def gen_rand_cond(self,batch_size=1):
        idx = np.random.choice(self.lab_vect.shape[0],size=batch_size)
        cond = self.lab_vect[idx]
        cond += np.random.uniform(-0.1, 0.1, cond.shape)
        return cond
        
    def _load_internally(self):
        print(">> loading "+str(self.dataset_name)+" ...") 
        
        if self.dataset_name == 'EmotioNet':
            labels = pd.read_csv(self.path_csv)
            labels.columns = [i.strip() for i in labels.columns]
            labels = labels[self.csv_columns]
            self.frame_list = labels.iloc[:,0].tolist()
            self.lab_vect = labels.iloc[:,1:].to_numpy()
            assert len(self.frame_list) == len(self.lab_vect)
            #labels.mean(axis=1)
        else:
            raise Exception("dataset not supported:"+str(self.dataset_name))
        
        n_images = min(len(self.frame_list),self.max_images) if self.max_images > 0 else len(self.frame_list)
        self.lab_vect = self.lab_vect[:n_images]
        self.img_vect = np.zeros((n_images,
                                 self.img_res[0],
                                 self.img_res[1],
                                 self.img_res[2]) , 'float32')
        
        
        for i in range(n_images):
            img_path = os.path.join(*[self.path_image_dir,self.image_patter_fn.replace('FRAME_ID',get_00000_num(self.frame_list[i]))])
            img = read_cv2_img(img_path)
            if self.normalize:
                #img = img/127.5 - 1.
                img = img/255 - 0.
            self.img_vect[i] = img

    def load_batch(self, batch_size=1, flip_prob=0, is_testing=False):
        if is_testing:
            raise Exception("not supported yet!")
        self.n_batches = int(len(self.img_vect) / batch_size)
        for i in range(self.n_batches):
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
    file_path = 'datasets/sample/'
    csv_filename = 'images.csv'
    images_dir = 'images_aligned' 
    base_path = os.path.abspath(os.path.dirname(file_path))
    csv_path = os.path.join(*[base_path,csv_filename])
    img_path = os.path.join(*[base_path,images_dir])
    ## 
    dl = InMemoryDataLoader(dataset_name='EmotioNet',
                            img_res=(112, 112,3),
                            path_csv=csv_path,
                            path_image_dir=img_path, 
                            max_images=12)
    ## 
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
    for batch_i, (labels , batch_images) in enumerate(dl.load_batch(batch_size=1)):
        pass
    
    
    
        
         
