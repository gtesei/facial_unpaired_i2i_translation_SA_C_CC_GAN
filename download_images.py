# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:46:20 2019

@author: gtesei
"""
import pandas as pd 
import requests
import os
import argparse
import time 
from os import listdir
from os.path import isfile, join

def process_data_file(filename,args,img_dir='EmotioNet'):
    #
    data = pd.read_csv(join(args.datadir,filename), sep='\t')
    data.columns = ['URL', 'URL_orig' ] + ['AU'+str(i) for i in range(1,61)]
    # 
    prefix = os.path.splitext(filename)[0]
    directory = join(args.imagedir,prefix,img_dir) 
    if not os.path.exists(directory):
        os.makedirs(directory)
    #
    for i in range(len(data)):
        Picture_request = None 
        print(i,data.loc[i,'URL'])
        try:
            Picture_request = requests.get(data.loc[i,'URL'],verify=False)
        except:
            print("Error connection --> trying URL_orig ... ") 
            try: 
                Picture_request = requests.get(data.loc[i,'URL'],verify=False)
            except: 
                print("Error again --> skipping ....") 
        if Picture_request.status_code == 200:
            with open(directory+"/"+"000"+str(i)+".jpg", 'wb') as f:
                f.write(Picture_request.content)
        if args.sample_size > 0 and i >= args.sample_size:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download EmotioNet')
    parser.add_argument('-datadir', help='sample data file', dest='datadir', type=str, default='../emotioNet_challenge_files_server_challenge_1.2_aws/')
    parser.add_argument('-imagedir', help='image directory', dest='imagedir', type=str, default='images')
    parser.add_argument('-sample_size', help='sample size', dest='sample_size', type=int, default=-1)
    parser.add_argument('-n_files', help='number of files to process', dest='n_files', type=int, default=22)
    parser.add_argument('-offset', help='number of files to skip', dest='offset', type=int, default=7)
    args = parser.parse_args()
    # print parameters
    print('-' * 30)
    print('Parameters .')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)
    start_time = time.time()
    # 
    dfs = [f for f in listdir(args.datadir) if isfile(join(args.datadir, f))]
    for i in range(args.n_files):
        filename = dfs[i]
        print(i,"=============>",filename)
        if i > args.offset-1:
        	process_data_file(filename,args)
        else:
        	print("..skipping ....")
    # 
    seconds = time.time() - start_time
    mins = seconds / 60
    hours = mins / 60
    days = hours / 24
    print("------>>>>>>> elapsed seconds: " + str(seconds))
    print("------>>>>>>> elapsed minutes: " + str(mins))
    print("------>>>>>>> elapsed hours: " + str(hours))
    print("------>>>>>>> elapsed days: " + str(days))