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

from scipy.fftpack import fft, ifft

from dtw import dtw
from fastdtw import fastdtw
import time
from scipy.spatial.distance import euclidean
from tslearn.clustering import GlobalAlignmentKernelKMeans
import loading_data
from loading_data import load_real_data, load_standardized_data,load_train_data,load_train_data_V
from scipy import stats
from sklearn.ensemble import IsolationForest
import seaborn as sns; sns.set()
#%%
# =============================================================================
# =============================================================================
# # take out the event pointers from any kind of model
# =============================================================================
# =============================================================================
dir='figures/all_events/'
event_points={}
for i in [0]:
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
    
    
    print(i)
    #%%
# =============================================================================
# =============================================================================
# # save event data from real and standardized and the reduced mean as well
# =============================================================================
# =============================================================================
data_file='data/Armin_Data/July_03/pkl/J3.pkl'
std_data=load_standardized_data(data_file)
#%%
r_data=load_real_data(data_file)
scale=20
shift=120

real_data={}
std_no_mean_data={}
standard_data={}
for i in event_points[3]['GAN_total_events']:
    i=int(i)
    start=scale*i-shift
    end=scale*i+shift
    tempreal=r_data[:,start:end]
    tempdata=std_data[:,start:end]
    real_data[i]=tempreal
    standard_data[i]=tempdata
    tempdata=(tempdata-tempdata.mean(axis=1).reshape(-1,1))
    std_no_mean_data[i]=tempdata
    
#%%
# =============================================================================
# =============================================================================
# #     save all type of events data for July third
# =============================================================================
# =============================================================================
real="figures/all_events/July_03/real.pkl"
std="figures/all_events/July_03/std.pkl"
stdnomean="figures/all_events/July_03/stdnomean.pkl"
output = open(real, 'wb')
pkl.dump(real_data, output)
output.close()

output = open(std, 'wb')
pkl.dump(standard_data, output)
output.close()
    

output = open(stdnomean, 'wb')
pkl.dump(std_no_mean_data, output)
output.close()

#%%
# =============================================================================
# =============================================================================
# # laod the event point
# =============================================================================
# =============================================================================
stdnomean="figures/all_events/July_03/stdnomean.pkl"

pkl_file = open(stdnomean, 'rb')
std_no_mean_data = pkl.load(pkl_file)
pkl_file.close()
#%%
# =============================================================================
# =============================================================================
# # selected event points for testifg clustering methods
# =============================================================================
# =============================================================================
selected_events=[350,351,11158,7417,21809,62447,42498,54563,66279,102488,103869
                 ,103860,103871,105156,69018,57959,56316,309485,306447,295168
                 ,255848,348846,348898,349143,349524,30855,28396,148978,49131,64830
                 ,77780,67276,121772,400302]
ii=0
for anom in classes[0]:
#    print(corr[14][ii])
    ii+=1
    plt.subplot(221)
    for i in [0,1,2]:
        plt.plot(std_no_mean_data[anom][i])
    plt.legend('A' 'B' 'C')
    plt.title('V')
        
    plt.subplot(222)
    for i in [3,4,5]:
        plt.plot(std_no_mean_data[anom][i])
    plt.legend('A' 'B' 'C')
    plt.title('I')  
    
    plt.subplot(223)
    for i in [6,7,8]:
        plt.plot(std_no_mean_data[anom][i])
    plt.legend('A' 'B' 'C') 
    plt.title('P')    
    
    plt.subplot(224)
    for i in [9,10,11]:
        plt.plot(std_no_mean_data[anom][i])
    plt.legend('A' 'B' 'C')
    plt.title('Q')    

    plt.show()
    
#%%
# =============================================================================
# =============================================================================
# # test the dtw for selected events
# =============================================================================
# =============================================================================
    
euclidean_norm = lambda x, y: np.abs(x - y)
#d, cost_matrix, acc_cost_matrix, path = dtw(standard_data[350][0], standard_data[309485][0], dist=euclidean_norm)
#plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
#
#plt.plot(path[0], path[1], 'w')
#plt.show()
#%%
dtw_dists=[]
for i in selected_events: 
    print(i)
    temp_dist=[]
    for j in selected_events:
#        distance, path = fastdtw(standard_data[i][3], standard_data[j][3], dist=euclidean)
        distance=np.sum(euclidean_norm(std_no_mean_data[i][0], std_no_mean_data[j][0]))
        temp_dist.append(distance)
    temp_dist=np.array(temp_dist)
    dtw_dists.append(temp_dist)
dtw_dists=np.array(dtw_dists)

#%%
ax = sns.heatmap(corr)
 
#%%
N=len(std_no_mean_data.keys())
N=selected_random_events.shape[0]
corr=np.zeros((N,N))
for idx1,anom1 in enumerate(selected_random_events):
    if idx1% 10==0:
        print('iter num: %i', idx1)
    for idx2,anom2 in enumerate(selected_random_events):
        if idx2>=idx1:
            if idx2% 10==0:
                print('iter num: %i', idx2)
            max_corr=0
            for i in range(120):
                cr=0
                for j in range(4):
                    cr+=np.corrcoef(std_no_mean_data[anom1][j*3],np.roll(std_no_mean_data[anom2][j*3],i-60))[0,1]
                cr=cr/4
                if cr>max_corr:
                    max_corr=cr
            corr[idx1,idx2]=max_corr
        else:
            corr[idx1,idx2]=corr[idx2,idx1]
#%%
events=np.array(list(std_no_mean_data.keys()))
evt_num=events.shape[0]
random_select=np.random.choice(evt_num, 200, replace=False)    
selected_random_events=events[random_select]
#%%
s=[]
for id,h in enumerate(correlation200[20]):
    if h> 0.7:
        s.append(selected_random_events[id])
#%%
# =============================================================================
# =============================================================================
# # clustering by eliminating similar ones
# =============================================================================
# =============================================================================
trh=0.7
classes={}
count=0
remain=corr.shape[0]
while remain>2:
    ax = sns.heatmap(corr)
#    plt.plot(ax)
    classes[count]=[]
    del_ids=[]
    rows=list(np.arange(1,remain+1)-1)
    for id,h in enumerate(corr[0]):
        if h> 0.7:
            classes[count].append(selected_random_events[id])
            del_ids.append(id)
            rows.remove(id)
    for i in del_ids:
        for id,h in enumerate(corr[i]):
            if h> 0.7:
                if not selected_random_events[id] in classes[count]:
                    classes[count].append(selected_random_events[id])
                    if id in rows:
                        rows.remove(id)
    count+=1
    corr=corr[rows][:,rows]
    remain=corr.shape[0]
    plt.show()

    
#%%
# =============================================================================
# =============================================================================
# # pick the candidate of a cluster
# =============================================================================
# =============================================================================
def pick_the_candidate(group):
    
    max=0
    if len(group)>1:
        for i in group:
            
            idx1=list(selected_random_events).index(i)
            temp=0
            for j in group:
                idx2=list(selected_random_events).index(j)
                if not i==j:
                    temp+=corr[idx1,idx2]
            print(temp)
            if max<temp:
                
                max=temp
                candidate=i
    else:
        candidate=group

    return candidate
#%%
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    