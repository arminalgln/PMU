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
import random
import pickle as pkl
import operator
import math
from sklearn import preprocessing
from keras.models import load_model
import time
from scipy.stats import norm
from scipy.io import loadmat
from natsort import natsorted
from scipy import stats
from seaborn import heatmap
import loading_data
from loading_data import load_real_data, load_standardized_data,load_train_data,load_train_data_V,load_train_vitheta_data_V,load_data_with_features,load_standardized_data_with_features
#%%
# =============================================================================
# =============================================================================
# =============================================================================
# # # extract candidate for the clusters  which extraxted by hand from July 03
# =============================================================================
# =============================================================================
# =============================================================================

cluster_folder_name='onedayclusters'
cluster_folder=os.listdir(cluster_folder_name)
separarted_events={}
cl_num=0
for cluster in cluster_folder:
    separarted_events[cl_num]=[]
    events=os.listdir(cluster_folder_name+'/'+cluster)
    for ev in events:
        separarted_events[cl_num].append(int(ev.split('.')[0]))
        
    cl_num+=1
#%%
# =============================================================================
# =============================================================================
## call data which includes V, I and theta (9 features)
# =============================================================================
# =============================================================================
filename='data/Armin_Data/July_03/pkl/julseppf3.pkl'
k=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG','TA', 'TB', 'TC']
#%%
# =============================================================================
# =============================================================================
# # standardized data
# =============================================================================
# =============================================================================
dds=load_standardized_data_with_features(filename,k)
#%%
# =============================================================================
# =============================================================================
# # normal data
# =============================================================================
# =============================================================================
dd=load_data_with_features(filename,k)
#%%
# =============================================================================
# =============================================================================
# # train data
# =============================================================================
# =============================================================================
start,SampleNum,N=(0,40,500000)
filename='data/Armin_Data/July_03/pkl/julseppf3.pkl'
k=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG','TA', 'TB', 'TC']
tt=load_train_vitheta_data_V(start,SampleNum,N,filename,k)
#%%        
# =============================================================================
# =============================================================================
# # max corr coeff funciton based on each two event
# =============================================================================
# =============================================================================
def ccf(anom1,anom2,data):
# =============================================================================
#     480 time duration for each event
# =============================================================================
    scale=6
    shift=0   
    SampleNum=40
    max_corr=-1
    for i in range(120):
        cr=0
        for j in range(9):
            cr+=np.corrcoef(data[j][anom1*int(SampleNum/2)-40*scale+shift:(anom1*int(SampleNum/2)+40*scale+shift)],np.roll(data[j][anom2*int(SampleNum/2)-40*scale+shift:(anom2*int(SampleNum/2)+40*scale+shift)],i-60))[0,1]            
        cr=cr/9
        if cr>max_corr:
            max_corr=cr
    return max_corr
#%%
# =============================================================================
# =============================================================================
# # Training model - extract candidate for each pre selected cluster
# =============================================================================
# =============================================================================
      

def candidate_correlation(cluster_events,data):   
    #select number of events that we want to consider in each group for training
    N=len(cluster_events)
    trh=50
    N=min(N,trh)
    corr=np.zeros((N,N))
    #restricted candidate

    selected_events=np.random.choice(cluster_events, N, replace=False)
    for idx1,anom1 in enumerate(selected_events):
        print(idx1)
#        if idx1% 100==0:
#            print('iter num: %i', idx1)
        tic=time.clock()
        for idx2,anom2 in enumerate(selected_events):
            if idx2>=idx1:
#                if idx2% 100==0:
#                    print('iter num: %i', idx2)
                max_corr=ccf(anom1,anom2,data)
                corr[idx1,idx2]=max_corr
            else:
                corr[idx1,idx2]=corr[idx2,idx1]
        toc = time.clock()
        print(toc-tic)
    
    index=np.argmax(sum(corr))
    
    candid=selected_events[index]
    
    return corr,candid

#%%
# =============================================================================
# =============================================================================
# # calculate candidate of each cluster
# =============================================================================
# =============================================================================
representatives={}
for cl in separarted_events:
    
    cluster_event=separarted_events[cl]
    _,representatives[cl]=candidate_correlation(separarted_events[cl],dds)

#%%
# =============================================================================
# =============================================================================
# # show representatives
# =============================================================================
# =============================================================================
for can in representatives:
    show([representatives[can]])
#%%
# =============================================================================
# =============================================================================
# =============================================================================
# # # test the whole events one by one to see the accuracy of the candidates    
# =============================================================================
# =============================================================================
# =============================================================================
test_event_clusters={}

for cl in separarted_events:
    print(cl)
    temp_cluster_evevnts=separarted_events[cl]
    #check with the representative
    count=0
    for event in temp_cluster_evevnts:
        if count<100:
            print(event)
            nearest_distance=-1
            for can in representatives:
                dist=ccf(event,representatives[can],dds)
                if dist>nearest_distance:
                    nearest_distance=dist
                    closest_candidate=can
            test_event_clusters[event]=[closest_candidate,cl]
            count+=1
        

#%%
# =============================================================================
# =============================================================================
# # calculate the accuracy of the models (building multiclass confusion matrix)
# =============================================================================
# =============================================================================
            
# =============================================================================
# =============================================================================
# #             whole clusters even with the one phase events
# =============================================================================
# =============================================================================
cl_cum=len(separarted_events)
confusion_matrix=np.zeros((cl_num,cl_cum))

for cl in separarted_events:
    print(cl)
    temp_cluster_evevnts=separarted_events[cl]
    #check with the representative
    count=0
    for event in temp_cluster_evevnts:
        if count<100:
            confusion_matrix[test_event_clusters[event][1],test_event_clusters[event][0]]+=1
        count+=1  

acc={}
acc['tp']=[]
acc['fp']=[]
acc['fn']=[]
acc['tn']=[]
for i in range(cl_num):
    acc['tp'].append(confusion_matrix[i,i])
    acc['fp'].append(sum(confusion_matrix[:,i])-confusion_matrix[i,i])
    acc['fn'].append(sum(confusion_matrix[i,:])-confusion_matrix[i,i])
    acc['tn'].append(sum(sum(confusion_matrix[:,:]))-acc['tp'][i]-acc['fp'][i]-acc['fn'][i])

#%%
# =============================================================================
# =============================================================================
# #     total accuracy of clustering model
# =============================================================================
# =============================================================================
total_acccuracy=(sum(acc['tp'])+sum(acc['tn']))/(sum(acc['tp'])+sum(acc['tn'])+sum(acc['fp'])+sum(acc['fn']))
#%%
    
# =============================================================================
# =============================================================================
# # Show each event we want from V, I and theta data
# =============================================================================
# =============================================================================
select_1224=dds
def show(events):
    SampleNum=40
    for anom in events:
            anom=int(anom)
            anom=events[anom]
            print(anom)
            
            plt.subplot(221)
            for i in [0,1,2]:
                plt.plot(select_1224[i][anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)])
#            plt.legend('A' 'B' 'C')
            plt.title('V')
                
            plt.subplot(222)
            for i in [3,4,5]:
                plt.plot(select_1224[i][anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)])
#            plt.legend('A' 'B' 'C')
            plt.title('I')  
            
            plt.subplot(223)
            for i in [6,7,8]:
                plt.plot(select_1224[i][anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)])
#            plt.legend('A' 'B' 'C') 
            plt.title('T')    
             
            plt.show()










