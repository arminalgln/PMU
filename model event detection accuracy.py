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

from scipy.fftpack import fft, ifft

from dtw import dtw
from fastdtw import fastdtw
import time
from scipy.spatial.distance import euclidean
from tslearn.clustering import GlobalAlignmentKernelKMeans
import xlrd

#%%

# =============================================================================
# =============================================================================
# # Read the event files for each model
# =============================================================================
# =============================================================================
dir='figures/all_events/'
event_points={}
events_acc_detail={}
for i in range(4):
    file=dir+'July_0'+str(i+3)
    GAN_events_file=file+'/GAN/anoms_july_0'+str(i+3)+'.csv'
    GAN_voltage_events_file=file+'/GAN_voltage/anoms_voltage_july_0'+str(i+3)+'.csv'
    Window_events_file=file+'/window/anoms_july_0'+str(i+3)+'.csv'
    
    GAN=pd.read_csv(GAN_events_file,header=None)[0].values
    GANV=pd.read_csv(GAN_voltage_events_file,header=None)[0].values
    window=pd.read_csv(Window_events_file,header=None)[0].values
    
    
    GAN_events_file=file+'/no_event'+'.xlsx'
    GAN_voltage_events_file=file+'/no_event_v'+'.xlsx'
    
    GANN=pd.read_excel(GAN_events_file)
    GANVN=pd.read_excel(GAN_voltage_events_file)
    GANVN=GANVN['GAN voltage'].values
    windowN=GANN['window'].values
    GANN=GANN['GAN'].values
    
    GANN = GANN[~np.isnan(GANN)]
    GANVN = GANVN[~np.isnan(GANVN)]
    windowN = windowN[~np.isnan(windowN)]

    event_points[i+3]={}
    event_points[i+3]['GAN_event']=np.setdiff1d(GAN,GANN)
    event_points[i+3]['GANV_event']=np.setdiff1d(GANV,GANVN)
    event_points[i+3]['GANV_total']=np.union1d(GAN,GANV)
    event_points[i+3]['GAN_total_events']=np.union1d(event_points[i+3]['GAN_event'],event_points[i+3]['GANV_event'])
    event_points[i+3]['window_event']=np.setdiff1d(window,windowN)
        
    all_event_points=[]
    for event in event_points[i+3]['GAN_total_events']:
    #    points=np
        low=event*20-240
        high=event*20+240
        rng=np.arange(low,high)
        all_event_points.append(rng)
        
        
    all_event_points =np.array(all_event_points)
    
    mutual_GAN_window=[]
    for j in event_points[i+3]['window_event']:
        if j in all_event_points:
            mutual_GAN_window.append(j)
    mutual_GAN_window=np.array(mutual_GAN_window)
    
    event_points[i+3]['mutual_GAN_window']=mutual_GAN_window
    
    whole_event_number=event_points[i+3]['GAN_total_events'].shape[0]+event_points[i+3]['window_event'].shape[0]-mutual_GAN_window.shape[0]
    
    
    events_acc_detail[i+3]={}
    
    events_acc_detail[i+3]['whole_detected_number']=event_points[i+3]['GANV_total'].shape[0]+window.shape[0]-mutual_GAN_window.shape[0]
    events_acc_detail[i+3]['whole_event_number']=whole_event_number
    TP=events_acc_detail[i+3]['GAN_TP']=event_points[i+3]['GAN_total_events'].shape[0]
    FP=events_acc_detail[i+3]['GAN_FP']=event_points[i+3]['GANV_total'].shape[0]-event_points[i+3]['GAN_total_events'].shape[0]
    FN=events_acc_detail[i+3]['GAN_FN']=whole_event_number-event_points[i+3]['GAN_total_events'].shape[0]
    TN=events_acc_detail[i+3]['GAN_TN']=events_acc_detail[i+3]['whole_detected_number']-(events_acc_detail[i+3]['GAN_TP']+events_acc_detail[i+3]['GAN_FP']+events_acc_detail[i+3]['GAN_FN'])
    events_acc_detail[i+3]['GAN_accuracy']=(TP+TN)/(TP+TN+FP+FN)
    events_acc_detail[i+3]['GAN_F1score']=(2*TP)/(2*TP+FP+FN)
    events_acc_detail[i+3]['GAN_MCC']=((TP*TN)-(FP*FN))/np.sqrt((TP+FP)*(TP+FN)*(TN*FP)*(TN*FN))
    
    
    TP=events_acc_detail[i+3]['W_TP']=event_points[i+3]['window_event'].shape[0]
    FP=events_acc_detail[i+3]['W_FP']=windowN.shape[0]
    FN=events_acc_detail[i+3]['W_FN']=whole_event_number-event_points[i+3]['window_event'].shape[0]
    TN=events_acc_detail[i+3]['W_TN']=events_acc_detail[i+3]['whole_detected_number']-(events_acc_detail[i+3]['W_TP']+events_acc_detail[i+3]['W_FP']+events_acc_detail[i+3]['W_FN'])
    events_acc_detail[i+3]['W_accuracy']=(TP+TN)/(TP+TN+FP+FN)
    events_acc_detail[i+3]['W_F1score']=(2*TP)/(2*TP+FP+FN)
    events_acc_detail[i+3]['W_MCC']=((TP*TN)-(FP*FN))/np.sqrt((TP+FP)*(TP+FN)*(TN*FP)*(TN*FN))
    
    print(i)
#events_acc_detail[i+3][GAN]={}
#events_acc_detail[i+3][GAN][]
#print(event_points[i+3]['window_event'].shape)
#print(all_event_points.shape)27
    
    
    
#%%
G_TP=0
G_FP=0
G_FN=0
G_TN=0

W_TP=0
W_FP=0
W_FN=0
W_TN=0

for day in events_acc_detail:
    
    G_TP+=events_acc_detail[day]['GAN_TP']
    G_FP+=events_acc_detail[day]['GAN_FP']
    G_FN+=events_acc_detail[day]['GAN_FN']
    G_TN+=events_acc_detail[day]['GAN_TN']
    
    W_TP+=events_acc_detail[day]['W_TP']
    W_FP+=events_acc_detail[day]['W_FP']
    W_FN+=events_acc_detail[day]['W_FN']
    W_TN+=events_acc_detail[day]['W_TN']
    print(day)
    
events_acc_detail['all']={}
TP=G_TP
FP=G_FP
FN=G_FN
TN=G_TN    
events_acc_detail['all']['GAN_whole_Days_accuracy']=(TP+TN)/(TP+TN+FP+FN)
events_acc_detail['all']['GAN_whole_Days_F1score']=(2*TP)/(2*TP+FP+FN)
events_acc_detail['all']['GAN_whole_Days_MCC']=((TP*TN)-(FP*FN))/math.sqrt((TP+FP)*(TP+FN)*(TN*FP)*(TN*FN))

TP=W_TP
FP=W_FP
FN=W_FN
TN=W_TN    
events_acc_detail['all']['W_whole_Days_accuracy']=(TP+TN)/(TP+TN+FP+FN)
events_acc_detail['all']['W_whole_Days_F1score']=(2*TP)/(2*TP+FP+FN)
events_acc_detail['all']['W_whole_Days_MCC']=((TP*TN)-(FP*FN))/math.sqrt((TP+FP)*(TP+FN)*(TN*FP)*(TN*FN))
    
#%%

# =============================================================================
# =============================================================================
# # the ones are in the window but GAN did not extracted
# =============================================================================
# =============================================================================
WyGANn=np.setdiff1d(event_points[6]['window_event'],mutual_GAN_window)

#mutual_shifts=[]
#for u in mutual_GAN_window:
#    u=int(u)
#    low=u-240
#    high=u+240
#    rng=np.arange(low,high)
#    mutual_shifts.append(rng)
#GANyWn=np.setdiff1d(all_event_points,mutual_GAN_window)
##%%
#GANyWn=np.unique(np.floor(GANyWn/20))
#    
#%%
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
filename='data/Armin_Data/July_06/pkl/J6.pkl'
select_1224=load_real_data(filename)
#%%
start,SampleNum,N=(0,40,500000)

for point in WyGANn:
    
    print(point)
    point=int(point)
    
    plt.subplot(221)
    for i in [0,1,2]:
        plt.plot(select_1224[i][point-120:point+120])
    plt.legend('A' 'B' 'C')
    plt.title('V')
        
    plt.subplot(222)
    for i in [3,4,5]:
        plt.plot(select_1224[i][point-120:point+120])
    plt.legend('A' 'B' 'C')
    plt.title('I')  
    
    plt.subplot(223)
    for i in [6,7,8]:
        plt.plot(select_1224[i][point-120:point+120])
    plt.legend('A' 'B' 'C') 
    plt.title('P')    
    
    plt.subplot(224)
    for i in [9,10,11]:
        plt.plot(select_1224[i][point-120:point+120])
    plt.legend('A' 'B' 'C')
    plt.title('Q')    
    plt.show()
#        
    
    
    
    
    
    
    
    