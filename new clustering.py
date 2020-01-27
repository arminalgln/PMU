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
from loading_data import load_real_data, load_standardized_data,load_train_data,load_train_data_V,load_standardized_data_with_features,load_data_with_features
from scipy import stats
from sklearn.ensemble import IsolationForest
import seaborn as sns; sns.set()
#%%
# =============================================================================
# =============================================================================
# # selected 3phase for clustering, saved in the data file clustering
# =============================================================================
# =============================================================================
selected_events_for_clustering
    #%%
# =============================================================================
# =============================================================================
# # save event data from real and standardized and the reduced mean as well
# =============================================================================
# =============================================================================
data_file='data/Armin_Data/July_03/pkl/julseppf3.pkl'
features=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG','TA', 'TB', 'TC']
std_data=load_standardized_data_with_features(data_file,features)
#%%
# =============================================================================
# =============================================================================
# # saving data for events
# =============================================================================
# =============================================================================
r_data=load_data_with_features(data_file,features)
scale=20
shift=240
#%%
real_data={}
std_no_mean_data={}
standard_data={}
for i in sample_events:
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
real="figures/all_events/July_03/new_real_3ph.pkl"
std="figures/all_events/July_03/new_std_3ph.pkl"
stdnomean="figures/all_events/July_03/new_stdnomean_3ph.pkl"
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
stdnomean="figures/all_events/July_03/new_stdnomean_3ph.pkl"

pkl_file = open(stdnomean, 'rb')
std_no_mean_data = pkl.load(pkl_file)
pkl_file.close()
#%%
def showstd(events):
    for anom in events:
            anom=int(anom)
            print(anom)
            
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
            plt.title('T')    
            
            plt.subplot(224)
#            for i in [9,10,11]:
#                plt.plot(select_1224[i][anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)])
#            plt.legend('A' 'B' 'C')
#            plt.title('Q')    
            plt.show()

#%%

#%%
#considered_events=selected_events_for_clustering[0:100]
#%%
# =============================================================================
# =============================================================================
# # medoids searching method iteratively
# =============================================================================
# =============================================================================
def initial_medoids(class_number):
    
    medoids=np.random.choice(considered_events, class_number, replace=False) 
    return medoids
 #%%
def similarity(event1,event2):
    max_corr=-10
    for i in range(120):
        cr=0
        for j in range(3):
            cr+=np.corrcoef(std_no_mean_data[event1][j],np.roll(std_no_mean_data[event2][j],i-60))[0,1]
        cr=cr/3
#        print(cr)
        if cr>max_corr:
            max_corr=cr

    sim=max_corr
    return sim
#%%
def cluster_assigned(old_medoids):
    new_clusters={}
    sum_sims={}
    for med in old_medoids:
        sum_sims[med]=0
        new_clusters[med]=[]
    for event in considered_events:
        close=-10
#        assigend_cluster=0
        for med in old_medoids:
            sim=similarity(event,med)
            if sim>close:
                close=sim
                assigend_cluster=med
        sum_sims[assigend_cluster]+=close
        new_clusters[assigend_cluster].append(event)
    
    return new_clusters,sum_sims
#%%
def new_med(new_cluster):
    N=len(new_cluster)
    corr=np.zeros((N,N))
#    print(N)
    for idx1,event1 in enumerate(new_cluster):
#        if idx1% 100==0:
#            print('iter num: %i', idx1)
#        tik=time.clock()
        for idx2,event2 in enumerate(new_cluster):
            if idx2>=idx1:
#                if idx2% 100==0:
#                    print('iter num: %i', idx2)
                
                corr[idx1,idx2]=similarity(event1,event2)
            else:
                corr[idx1,idx2]=corr[idx2,idx1]  
    
    col_sum=np.sum(corr,axis=0)
    
    new_med_index=np.max(col_sum)
    event_list=list(col_sum)
    new_med_index=event_list.index(np.max(new_med_index))
    
    new_medoid=new_cluster[new_med_index]
    
    return new_medoid
#%%
# =============================================================================
# finding the best medoids based on cluster numbers
# =============================================================================
considered_events=np.random.choice(selected_events_for_clustering, 400, replace=False)    
crt=0
class_number=6
init_medoids=initial_medoids(class_number)
first_step=0
iter=0
objective=[]
while crt==0:
    print(iter)
    if first_step==0:
        old_medoids=init_medoids
        new_medoids=init_medoids
        first_step=1
        
    new_clusters,sum_sims=cluster_assigned(old_medoids)        
   
    
    objective.append(sum(sum_sims.values()))
    iter+=1
    
     
    new_medoids=[]
    for cluster in new_clusters:
        new_medoids.append(new_med(new_clusters[cluster]))
        
    
    print(new_medoids,old_medoids)
    
    nm=[]
    for i in new_medoids:
        nm.append(int(i))
    nm.sort()
        
    om=[]
    for i in old_medoids:
        om.append(int(i))
    om.sort()    
    
    count=0
    for i in om:
        if i in nm:
            count+=1
            
    print(count)
    if count==class_number:
        crt=1
    
    old_medoids=new_medoids
 
 #%%
# =============================================================================
# sample data correlation matrix 
# =============================================================================
sample_shape=considered_events.shape[0]
selected_corr=np.zeros((sample_shape,sample_shape))
 
#%%
# =============================================================================
# =============================================================================
# # Event clusters extracting from folder
# =============================================================================
# =============================================================================
clusters={}
clusters_together=[]
for i in os.listdir('clusters'):
    clusters[i]=[]
    for e in os.listdir('clusters/'+i):
        clusters[i].append(e.split('.')[0])
        clusters_together.append(e.split('.')[0])

#%%
sample_events=['350','351','3182','4743','7419','49465',
               '57881','67737','69018','88255','254519',
               '127594','144417','12901','254742','12914','13130','26959','30703',
               '496291']
sample_events=clusters_together

sample_events=np.array(sample_events)
sample_events_int=[int(x) for x in sample_events]

#%%
real_data={}
std_no_mean_data={}
standard_data={}
for i in sample_events:
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
# correlation function
# =============================================================================

def event_corr(sample_events_int,std_no_mean_data):
#    N=len(std_no_mean_data.keys())
    N=sample_events.shape[0]
    corr=np.zeros((N,N))
    for idx1,anom1 in enumerate(sample_events_int):
        if idx1% 100==0:
            print('iter num: %i', idx1)
        tik=time.clock()
        for idx2,anom2 in enumerate(sample_events_int):
            if idx2>=idx1:
                if idx2% 100==0:
                    print('iter num: %i', idx2)
                max_corr=0
                for i in range(120):
                    cr=0
                    for j in range(9):
                        cr+=np.corrcoef(std_no_mean_data[anom1][j],np.roll(std_no_mean_data[anom2][j],i-60))[0,1]
                    cr=cr/9
                    if cr>max_corr:
                        max_corr=cr
                corr[idx1,idx2]=max_corr
            else:
                corr[idx1,idx2]=corr[idx2,idx1]
        toc = time.clock()
        print(toc-tik)

    return corr
#%%
# =============================================================================
# correlation of the selected sample events
# =============================================================================
   
sample_corr=event_corr(sample_events_int,std_no_mean_data)
#%%
# =============================================================================
# =============================================================================
# # save samples and corr in order to pass to the matlab
# =============================================================================
# =============================================================================
    

import numpy as np
import scipy.io


scipy.io.savemat('correvent179.mat', dict(corr=sample_corr, events=sample_events_int))























    
    
    
    