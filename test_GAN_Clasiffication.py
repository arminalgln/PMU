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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from dtw import dtw
from fastdtw import fastdtw
#%% 
   
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

#%%
data_files=os.listdir('data/Armin_Data')
event_points={}
start,SampleNum,N=(0,40,500000)
for day in anomalies:
    print(day)
    anoms=anomalies[day]
    dir="data/Armin_Data/"+ day + "/pkl/"
    selectedfile=os.listdir(dir)[0]
    filename = dir + selectedfile
    select_1224=load_standardized_data(filename)
    event_points[day]={}
    for anom in anoms:
        anom=int(anom)
        event_points[day][anom]=select_1224[0:12,anom*int(SampleNum/2)-120:(anom*int(SampleNum/2)+120)]
#%%
# =============================================================================
# =============================================================================
# #         save the anomalies standardized data for 15 days
# =============================================================================
# =============================================================================
anomcsvfile="data/Armin_Data/anomsknnformat.pkl"
output = open(anomcsvfile, 'wb')
pkl.dump(event_points, output)
output.close()
#%%
anomcsvfile="data/Armin_Data/anomsknnformat.pkl"
pkl_file = open(anomcsvfile, 'rb')
event_points = pkl.load(pkl_file)
pkl_file.close()
#%%
X=[]
for day in event_points:
    for event in event_points[day]:
        X.append(event_points[day][event].ravel())
X=np.array(X)
#%%
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
#%%

for n_clusters in np.arange(10,40):
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

#%%
#pkl_file = open(anomcsvfile, 'rb')
#test = pkl.load(pkl_file)
#pkl_file.close()
#%%
similarity_matrix=[]
similarity_scores={}
tik=time.clock()
for day1 in event_points:
    similarity_scores[day1]={}
    print(day1)
    for anom1 in event_points[day1]:
        print(anom1)
        temp_similarity=[]
        
        similarity_scores[day1][anom1]={}
        
        x1=event_points[day1][anom1][::3]-np.mean(event_points[day1][anom1][::3],axis=1).reshape(4,1)
        x1=x1.ravel()
        
        for day2 in event_points:
            print(day2)
            similarity_scores[day1][anom1][day2]={}
            
            for anom2 in event_points[day2]:
                print(anom2)
                x2=event_points[day2][anom2][::3]-np.mean(event_points[day2][anom2][::3],axis=1).reshape(4,1)
                x2=x2.ravel()

#        plt.plot(event_points['July_10'][i][0]-np.mean(event_points['July_10'][i][0]))
#        plt.plot(event_points['July_10'][j][0]-np.mean(event_points['July_10'][j][0]))
#        plt.show()
                d, path = fastdtw(x1, x2, dist=euclidean_norm)
                print(d)
                similarity_scores[day1][anom1][day2][anom2]=d
                temp_similarity.append(d)
                
        temp_similarity=np.array(temp_similarity)
        similarity_matrix.append(temp_similarity)
similarity_matrix=np.array(similarity_matrix)
toc = time.clock()
print(toc-tik)
time_4features=toc-tik
#        print(d)
#        plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
#        plt.plot(path[0], path[1], 'w')
#        plt.show()
#        print('...........................................................')
#%%
fft_scores={}
total_events=0
for day1 in event_points:
    fft_scores[day1]={}
#    print(day1)

    for count,anom1 in enumerate(event_points[day1]):
#        print(anom1)
        total_events+=1
        x1=event_points[day1][anom1][::3]-np.mean(event_points[day1][anom1][::3],axis=1).reshape(4,1)      
        
        fft_scores[day1][anom1]=np.concatenate((np.fft.fftn(x1)[:,0:120].real.ravel(),np.fft.fftn(x1)[:,0:120].imag.ravel()),axis=None)

        
        if count% 500==0:
            print('iter num: %count', count)
print(total_events)
anomcsvfile="data/Armin_Data/fftscores.pkl"
output = open(anomcsvfile, 'wb')
pkl.dump(fft_scores, output)
output.close()
#%%%
for day1 in ['July_03']:
    similarity_scores[day1]={}
    print(day1)
    for anom1 in event_points[day1]:
        temp_similarity=[]
        print(anom1)
        similarity_scores[day1][anom1]={}
        
        x1=event_points[day1][anom1][::3]-np.mean(event_points[day1][anom1][::3],axis=1).reshape(4,1)
        x1=x1[3]
        ff=np.fft.fft(x1)
        freq = np.fft.fftfreq(x1.shape[-1])
        
        widths = np.arange(1, 240)
        cwtmatr = signal.cwt(x1, signal.ricker,widths)
        plt.subplot(131)
        plt.plot(freq, ff.real, freq, ff.imag)
        plt.subplot(132)
        plt.plot(x1)
        plt.subplot(133)
        plt.imshow(cwtmatr, extent=[-1, 1, 31, 1], cmap='PRGn', aspect='auto',
              vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
        plt.show()

