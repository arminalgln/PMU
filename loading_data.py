import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras
from keras.layers import Dense, Dropout, Input, Embedding, LSTM, Reshape, CuDNNLSTM
from keras.models import Model,Sequential
from keras.datasets import mnist
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.optimizers import adam
import numpy as np
import tensorflow as tf

import pickle as pkl
import operator
import math
from sklearn import preprocessing
from keras.models import load_model
import time
from scipy.stats import norm
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.io import loadmat
from natsort import natsorted

from scipy.fftpack import fft, ifft

from dtw import dtw
from fastdtw import fastdtw
import time
from scipy.spatial.distance import euclidean
from tslearn.clustering import GlobalAlignmentKernelKMeans
# =============================================================================
# =============================================================================
# # standardized data extraxtion
# =============================================================================
# =============================================================================
#filename='data/Armin_Data/July_03/pkl/jul3.pkl'
def load_standardized_data(filename):
         #read a pickle file
         
    pmu='1224'

    pkl_file = open(filename, 'rb')
    selected_data = pkl.load(pkl_file)
    pkl_file.close()
    selected_data=pd.DataFrame(selected_data)
    selected_data=selected_data.fillna(method='ffill')
    print(selected_data.keys())
    data=selected_data[pmu]
    features=['L3MAG','L2MAG','L1MAG', 'C1MAG',
       'C2MAG', 'C3MAG', 'PA', 'PB', 'PC', 'QC', 'QB','QA']
    
    select=[]
    for f in features:
        select.append(list(data[f]))
    
    select=np.array(select)
    print(select.shape)
    select=preprocessing.scale(select,axis=1)
    
    
    return select

    
# =============================================================================
# =============================================================================
# # real data extraxtion
# =============================================================================
# =============================================================================
#filename='data/Armin_Data/July_03/pkl/jul3.pkl'
def load_real_data(filename):
         #read a pickle file
         
    pmu='1224'

    pkl_file = open(filename, 'rb')
    selected_data = pkl.load(pkl_file)
    pkl_file.close()
    selected_data=pd.DataFrame(selected_data)
    selected_data=selected_data.fillna(method='ffill')
    print(selected_data.keys())
    data=selected_data[pmu]
    features=['L3MAG','L2MAG','L1MAG', 'C1MAG',
       'C2MAG', 'C3MAG', 'PA', 'PB', 'PC', 'QC', 'QB','QA']
    
    select=[]
    for f in features:
        select.append(list(data[f]))
    
    select=np.array(select)
    
    
    return select

def load_train_data(start,SampleNum,N,filename):
         #read a pickle file
    pkl_file = open(filename, 'rb')
    selected_data = pkl.load(pkl_file)
    pkl_file.close()
    for pmu in ['1224']:
        selected_data[pmu]=pd.DataFrame.from_dict(selected_data[pmu])
    features=['L3MAG','L2MAG','L1MAG', 'C1MAG',
       'C2MAG', 'C3MAG', 'PA', 'PB', 'PC', 'QC', 'QB','QA']
    
    print(selected_data.keys())
    select=[]
    for f in features:
        select.append(selected_data[pmu][f])
    
    
    selected_data=0
    select=np.array(select)
    
    print(select.shape)
    select=preprocessing.scale(select,axis=1)
    
#    selected_data=0
    end=start+SampleNum
    shift=int(SampleNum/2)
    
    train_data=np.zeros((N,12,SampleNum))
#    reduced_mean=np.zeros((12,20))
    for i in range(N):
        if i% 1000==0:
            print('iter num: %i', i)
        temp=select[:,start+i*shift:end+i*shift] 
        temp=(temp-temp.mean(axis=1).reshape(-1,1)) ## reduced mean
#        temp = preprocessing.scale(temp,axis=1)  ## standardized
#        reduced_mean=np.concatenate((reduced_mean,temp[:,0:20]),axis=1)
        train_data[i,:]=temp
    
    
    # convert shape of x_train from (60000, 28, 28) to (60000, 784) 
    # 784 columns per row
    
    return train_data#,select,selected_data#,select_proc,reduced_mean
#X_train=load_data()
#print(X_train.shape)


def load_train_data_V(start,SampleNum,N,filename):
         #read a pickle file
    pkl_file = open(filename, 'rb')
    selected_data = pkl.load(pkl_file)
    pkl_file.close()
    for pmu in ['1224']:
        selected_data[pmu]=pd.DataFrame.from_dict(selected_data[pmu])
    features=['L1MAG','L2MAG', 'L3MAG']
    
    print(selected_data.keys())
    select=[]
    for f in features:
        select.append(selected_data[pmu][f])
    
    
    selected_data=0
    select=np.array(select)
    
    print(select.shape)
    select=preprocessing.scale(select,axis=1)
    
#    selected_data=0
    end=start+SampleNum
    shift=int(SampleNum/2)
    
    train_data=np.zeros((N,3,SampleNum))
#    reduced_mean=np.zeros((12,20))
    for i in range(N):
        if i% 1000==0:
            print('iter num: %i', i)
        temp=select[:,start+i*shift:end+i*shift] 
        temp=(temp-temp.mean(axis=1).reshape(-1,1)) ## reduced mean
#        temp = preprocessing.scale(temp,axis=1)  ## standardized
#        reduced_mean=np.concatenate((reduced_mean,temp[:,0:20]),axis=1)
        train_data[i,:]=temp
    
    
    # convert shape of x_train from (60000, 28, 28) to (60000, 784) 
    # 784 columns per row
    
    return train_data#,select,selected_data#,select_proc,reduced_mean
#X_train=load_data()
#print(X_train.shape)
    


    
def load_data_with_features(filename,features):
         #read a pickle file
         
    pmu='1224'

    pkl_file = open(filename, 'rb')
    selected_data = pkl.load(pkl_file)
    pkl_file.close()
    selected_data=pd.DataFrame(selected_data)
    selected_data=selected_data.fillna(method='ffill')
    print(selected_data.keys())
    data=selected_data[pmu]
    
    select=[]
    for f in features:
        select.append(list(data[f]))
    
    select=np.array(select)
    print(select.shape)
#    select=preprocessing.scale(select,axis=1)
    
    
    return select

def load_standardized_data_with_features(filename,features):
         #read a pickle file
         
    pmu='1224'

    pkl_file = open(filename, 'rb')
    selected_data = pkl.load(pkl_file)
    pkl_file.close()
    selected_data=pd.DataFrame(selected_data)
    selected_data=selected_data.fillna(method='ffill')
    print(selected_data.keys())
    data=selected_data[pmu]
    
    select=[]
    for f in features:
        select.append(list(data[f]))
    
    select=np.array(select)
    print(select.shape)
    select=preprocessing.scale(select,axis=1)
    
    
    return select

def load_train_vitheta_data_1225(start,SampleNum,N,filename,features):
         #read a pickle file
    pkl_file = open(filename, 'rb')
    selected_data = pkl.load(pkl_file)
    pkl_file.close()
    
    selected_data=pd.DataFrame.from_dict(selected_data)
#    features=['L1MAG','L2MAG', 'L3MAG']
    
    print(selected_data.keys())
    select=[]
    for f in features:
        select.append(selected_data[f])
    
    
    selected_data=0
    select=np.array(select)
    
    print(select.shape)
    select=preprocessing.scale(select,axis=1)
    
#    selected_data=0
    end=start+SampleNum
    shift=int(SampleNum/2)
    
    train_data=np.zeros((N,9,SampleNum))
#    reduced_mean=np.zeros((12,20))
    for i in range(N):
        if i% 1000==0:
            print('iter num: %i', i)
        temp=select[:,start+i*shift:end+i*shift] 
        temp=(temp-temp.mean(axis=1).reshape(-1,1)) ## reduced mean
#        temp = preprocessing.scale(temp,axis=1)  ## standardized
#        reduced_mean=np.concatenate((reduced_mean,temp[:,0:20]),axis=1)
        train_data[i,:]=temp
    
    
    # convert shape of x_train from (60000, 28, 28) to (60000, 784) 
    # 784 columns per row
    
    return train_data#,select,selected_data#,select_proc,reduced_mean


def load_train_vitheta_data_V(start,SampleNum,N,filename,features):
         #read a pickle file
    pkl_file = open(filename, 'rb')
    selected_data = pkl.load(pkl_file)
    pkl_file.close()
    for pmu in ['1224']:
        selected_data[pmu]=pd.DataFrame.from_dict(selected_data[pmu])
#    features=['L1MAG','L2MAG', 'L3MAG']
    
    print(selected_data.keys())
    select=[]
    for f in features:
        select.append(selected_data[pmu][f])
    
    
    selected_data=0
    select=np.array(select)
    
    print(select.shape)
    select=preprocessing.scale(select,axis=1)
    
#    selected_data=0
    end=start+SampleNum
    shift=int(SampleNum/2)
    
    train_data=np.zeros((N,9,SampleNum))
#    reduced_mean=np.zeros((12,20))
    for i in range(N):
        if i% 1000==0:
            print('iter num: %i', i)
        temp=select[:,start+i*shift:end+i*shift] 
        temp=(temp-temp.mean(axis=1).reshape(-1,1)) ## reduced mean
#        temp = preprocessing.scale(temp,axis=1)  ## standardized
#        reduced_mean=np.concatenate((reduced_mean,temp[:,0:20]),axis=1)
        train_data[i,:]=temp
    
    
    # convert shape of x_train from (60000, 28, 28) to (60000, 784) 
    # 784 columns per row
    
    return train_data#,select,selected_data#,select_proc,reduced_mean
#X_train=load_data()
#print(X_train.shape)
    



   #here we Import raw data for March 9th for all three PMUs and saved each pmu separately

# =============================================================================
#    ###Import raw data for MArch 9th for all three PMUs
# def all_4_PMU_data():
#     whole_data={}
#     dir = 'Raw_data/'
#     files = os.listdir(dir)
#     files = natsorted(files)
#     PMU=['1086','1224','1200','1225']
#     for p in PMU:
#         whole_data[p]={}
#     
#     for f in files:
#         print(f)
#         #print(dir+f)
#         temp_data=pd.read_csv(dir+f)
#         k=temp_data.keys()
#         
#         for key in k:
# #            print(key)
#             for p in PMU:
# #                print(p)
#                 if (p in key.split('/'))  :
#                     
# #                    print(key.split('/'))
# #                    print(key.split('/')[2].split(' ')[0])
#                     if (key.split('/')[2].split(' ')[1]=='(Mean)') and (key.split('/')[2].split(' ')[0]!='LSTATE'):
# #                        print(p)
#                         col=key.split('/')[2].split(' ')[0]
# #                        print(col)
#                         if col in whole_data[p]:
#                             
#                             whole_data[p][col]=np.append(whole_data[p][col],temp_data[key].values)
# #                            whole_data[p][col].append(list(temp_data[key].values))
#                             print(len(whole_data[p][col]))
#                         else:
#                             print(col)
#                             whole_data[p][col]=temp_data[key].values
#         
#         
#     return whole_data
# 
# #%%
# PMU=['1086','1224','1200','1225']
# for p in PMU:
#     dir = 'Raw_data/'
#     os.mkdir(dir+p)
#     output = open(dir+p+'/data', 'wb')
#     pkl.dump(whole[p], output)
#     output.close()
#     #%%
# 
# =============================================================================
