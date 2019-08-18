# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
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

#%% 
   
# =============================================================================
# =============================================================================
# # standardized data extraxtion
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
    features=['L1MAG','L2MAG', 'L3MAG','C1MAG',
       'C2MAG', 'C3MAG', 'PA', 'PB', 'PC', 'QA', 'QB', 'QC']
    
    select=[]
    for f in features:
        select.append(list(data[f]))
    
    select=np.array(select)
    print(select.shape)
    select=preprocessing.scale(select,axis=1)
    
    
    return select
#%%
    
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
    features=['L1MAG','L2MAG', 'L3MAG','C1MAG',
       'C2MAG', 'C3MAG', 'PA', 'PB', 'PC', 'QA', 'QB', 'QC']
    
    select=[]
    for f in features:
        select.append(list(data[f]))
    
    select=np.array(select)
    
    
    return select

#%%
# =============================================================================
# Reading the files in the data to make a for
# =============================================================================
files=os.listdir('figures/1224_15_days/')
#%%
anomalies={}
for file in files:
    dir='figures/1224_15_days/'
    dir=dir+file+"/GAN"
    tempfiles=os.listdir(dir)
    for f in tempfiles:
        if f.endswith(".csv"):
            anomfile=dir+'/'+f
            ta=pd.read_csv(anomfile)
            anomalies[file]=ta.values
#%%
# =============================================================================
# make a place to save all 1224 events data wrt each day, whether my method or Alirezas
# =============================================================================
dst="figures/1224_15_days"
os.mkdir(dst)
#%%
for file in ['July_14']:
    if file == 'July_03':
        continue
# =============================================================================
#     extract train data for the selected day
# =============================================================================
    print(file)
    start,SampleNum,N=(0,40,500000)
    dir="data/Armin_Data/"+ file + "/pkl/"
    selectedfile=os.listdir(dir)[0]
    filename = dir + selectedfile
    X_train= load_data(start,SampleNum,N,filename)
    #batch_count = X_train.shape[0] / batch_size
    
    X_train=X_train.reshape(N,12*SampleNum)
    X_train=X_train.reshape(N,SampleNum,12)
# =============================================================================
#     calculate the score for the selected day
# =============================================================================
    #a=discriminator.predict_on_batch(X_train)
    rate=1000
    shift=N/rate
    scores=[]
    for i in range(rate-1):
        temp=discriminator.predict_on_batch(X_train[int(i*shift):int((i+1)*shift)])
        scores.append(temp)
        print(i)
    
    scores=np.array(scores)
    scores=scores.ravel()


    probability_mean=np.mean(scores)
    a=scores-probability_mean

# =============================================================================
# obtain the boundaries for events
# =============================================================================
    zp=3
    
    data = a
# Fit a normal distribution to the data:
    mu, std = norm.fit(data)
    
    high=mu+zp*std
    low=mu-zp*std
    
    anoms_1224=np.union1d(np.where(a>=high)[0], np.where(a<=low)[0])
    print(anoms_1224.shape)
# =============================================================================
# select the real data for the day
# =============================================================================
    select_1224=load_real_data(filename)
# =============================================================================
# make file to save photos for the GAN model
# =============================================================================
    dst="figures/1224_15_days/"+file
    os.mkdir(dst)
    dst=dst+"/GAN"
    os.mkdir(dst)
# =============================================================================
#     save training number period as an events
# =============================================================================
    anomcsvfile=dst+"/anoms_"+file+".csv"
    np.savetxt(anomcsvfile, anoms_1224, delimiter=",")
    
    event_points=[]
    for anom in anoms_1224:
        print(anom)
        
        plt.subplot(221)
        for i in [0,1,2]:
            plt.plot(select_1224[i][anom*int(SampleNum/2)-120:(anom*int(SampleNum/2)+120)])
        plt.legend('A' 'B' 'C')
        plt.title('V')
            
        plt.subplot(222)
        for i in [3,4,5]:
            plt.plot(select_1224[i][anom*int(SampleNum/2)-120:(anom*int(SampleNum/2)+120)])
        plt.legend('A' 'B' 'C')
        plt.title('I')  
        
        plt.subplot(223)
        for i in [6,7,8]:
            plt.plot(select_1224[i][anom*int(SampleNum/2)-120:(anom*int(SampleNum/2)+120)])
        plt.legend('A' 'B' 'C') 
        plt.title('P')    
        
        plt.subplot(224)
        for i in [9,10,11]:
            plt.plot(select_1224[i][anom*int(SampleNum/2)-120:(anom*int(SampleNum/2)+120)])
        plt.legend('A' 'B' 'C')
        plt.title('Q')    
        figname=dst+"/"+str(anom)
        plt.savefig(figname)
        plt.show()
# =============================================================================
# find the wide range of anomalies point to compare with Alirezas data    
# =============================================================================
        low=anom*20-120
        high=anom*20+120
        rng=np.arange(low,high)
        event_points.append(rng)
    event_points=np.array(event_points).ravel()
    


    # =============================================================================
    # =============================================================================
    # # read pointers from matlab file: (Alireza's results)
    # =============================================================================
    # =============================================================================
    
    pointers = loadmat('data/pointer.mat')
    pf='Jul'+"_"+file.split('_')[1]
    points=pointers['pointer'][pf][0][0].ravel()
    points.sort()
    
    
# =============================================================================
# common anomalies GAN and window
# =============================================================================
    common_anoms=np.intersect1d(points,event_points)
    dst="figures/1224_15_days/"+file
    anomcsvfile=dst+"/common"+file+".csv"
    np.savetxt(anomcsvfile, common_anoms, delimiter=",")
# =============================================================================
# make folder to save Alirezas event in the same day
# =============================================================================
    dst="figures/1224_15_days/"+file
    dst=dst+"/window"
    os.mkdir(dst)
# =============================================================================
#     save the window method event points
# =============================================================================
    anomcsvfile=dst+"/anoms_"+file+".csv"
    np.savetxt(anomcsvfile, points, delimiter=",")

    for anom in points:
        print(anom)
        
        plt.subplot(221)
        for i in [0,1,2]:
            plt.plot(select_1224[i][anom-120:(anom+120)])
        plt.legend('A' 'B' 'C')
        plt.title('V')
            
        plt.subplot(222)
        for i in [3,4,5]:
            plt.plot(select_1224[i][anom-120:(anom+120)])
        plt.legend('A' 'B' 'C')
        plt.title('I')  
        
        plt.subplot(223)
        for i in [6,7,8]:
            plt.plot(select_1224[i][anom-120:(anom+120)])
        plt.legend('A' 'B' 'C') 
        plt.title('P')    
        
        plt.subplot(224)
        for i in [9,10,11]:
            plt.plot(select_1224[i][anom-120:(anom+120)])
        plt.legend('A' 'B' 'C')
        plt.title('Q')    
        figname=dst+"/"+str(anom)
        plt.savefig(figname)
        plt.show()