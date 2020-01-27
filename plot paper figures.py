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
from loading_data import load_real_data, load_standardized_data,load_train_data,load_train_data_V,load_data_with_features

from sklearn.ensemble import IsolationForest
#%%

filename='data/Armin_Data/July_03/pkl/J3.pkl'
start,SampleNum,N,filename=(0,40,500000,filename)
select_1224=load_real_data(filename)
#%%
filename='data/Armin_Data/July_03/pkl/rawdata3.pkl'
k=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG','L1Ang','L2Ang','L3Ang','C1Ang','C2Ang','C3Ang']
#%%
dds=load_standardized_data_with_features(filename,k)
#%%
dd=load_data_with_features(filename,k)

#%%
anom_select=[60613]

#anom_select=[350,351,3182,4743,7419,49465,57881,67737,69018,88255,254519,127594,144417,12901,254742,12914,13130,26959,30703,496291]

#anom_select=[36687, 37490, 41092, 54565, 66277, 84418, 85595, 322135, 338446, 425659, 354777,339351, 252725]
scale=8
shift=0
k=0
select_1224=dd
for anom in anom_select:
        k+=1
        print(anom)
        anom=int(anom)
        plt.subplot(221)
        for i in [2,1,0]:
            plt.plot(select_1224[i][anom*int(SampleNum/2)-40*scale+shift:(anom*int(SampleNum/2)+40*scale+shift)])
#        plt.legend('A' 'B' 'C',fontsize= 20,loc=6)
        plt.yticks(fontsize=15)
#        plt.ylim([7100,7230])
#        plt.figtext(.5,.9,'Temperature', fontsize=100, ha='center')
        plt.title('V (magnitude)',fontsize= 30)
        
            
        plt.subplot(222)
        for i in [3,4,5]:
            plt.plot(select_1224[i][anom*int(SampleNum/2)-40*scale+shift:(anom*int(SampleNum/2)+40*scale+shift)])
#        plt.legend('A' 'B' 'C')
        plt.title('V (Angle)',fontsize= 30)  
        plt.yticks(fontsize=15)
        
        plt.subplot(223)
        for i in [6,7,8]:
            plt.plot(select_1224[i][anom*int(SampleNum/2)-40*scale+shift:(anom*int(SampleNum/2)+40*scale+shift)]/1000)
#        plt.legend('A' 'B' 'C') 
        plt.title('I (Magnitude)',fontsize= 30)    
        plt.xlabel('Timeslots',fontsize= 30)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        
        plt.subplot(224)
        for i in [11,10,9]:
            plt.plot(select_1224[i][anom*int(SampleNum/2)-40*scale+shift:(anom*int(SampleNum/2)+40*scale+shift)]/1000)
#        plt.legend('A' 'B' 'C')
        plt.title('I (Angle)',fontsize= 30)
        plt.xlabel('Timeslots',fontsize= 30)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
#        plt.savefig('event.pdf', format='pdf')
#        figname='figures/paper/huge_osc.pdf'
#        plt.savefig(figname)
        plt.show()#%%
        
        #%%
# =============================================================================
# jsut GAN scores
# =============================================================================
plt.scatter(whole_features['scores_V'], whole_features['scores'],color=whole_features['color'])
#plt.legend('Noraml' 'Events',fontsize= 20,loc=6)
plt.yticks(fontsize=15)
#        plt.figtext(.5,.9,'Temperature', fontsize=100, ha='center')
plt.xlabel('MPM',fontsize= 30)
plt.ylabel('MV',fontsize= 30)
#%%
# =============================================================================
# =============================================================================
# # all proposed model
# =============================================================================
# =============================================================================


    #%%
zp=3.1
anoms31={}
names=['scores','scores_V','maxvar','maxmaxmin']
for i,d in enumerate(data):
    dt = d
    # Fit a normal distribution to the data:
    mu, std = norm.fit(dt)
    
    high=mu+zp*std
    low=mu-zp*std
    anoms_1224=np.union1d(np.where(dt>=high)[0], np.where(dt<=low)[0])
    print(anoms_1224.shape)
    anoms31[names[i]]=anoms_1224

#%%
t1=np.union1d(anoms31['scores'],anoms31['scores_V'])
t2=np.union1d(anoms31['maxvar'],anoms31['maxmaxmin'])
total_events=np.union1d(t1,t2)


#%%
whole_features['new_anoms']=np.zeros((N,1))
for i in total_events:
    i=int(float(i))
    whole_features['new_anoms'][i]=1
#%%
an=0
whole_features['new_color']=[]
for i in whole_features['new_anoms']:
#    print(i)
    if int(i) == 0:
        whole_features['new_color'].append('b')
    else:
        an+=1
        whole_features['new_color'].append('r')
        
whole_features['new_color']=np.array(whole_features['new_color'])
print(an) 

#%%

plt.scatter(whole_features['maxmaxmin'], whole_features['maxvar'],color=whole_features['new_color'])
#plt.legend('Noraml' 'Events',fontsize= 20,loc=6)
plt.yticks(fontsize=15)
#        plt.figtext(.5,.9,'Temperature', fontsize=100, ha='center')
plt.xlabel('MPM',fontsize= 30)
plt.ylabel('MV',fontsize= 30)

#%%
# =============================================================================
# =============================================================================
# # proposed
# =============================================================================
# =============================================================================
total=3152
t_ev=2621
TP,TN,FP,FN=[2321,60,60,200]
acc=(TP+TN)/(TP+TN+FP+FN)
f1=(2*TP)/(2*TP+FP+FN)
mcc=((TP*TN)-(FP*FN))/np.sqrt((TP+FP)*(TP+FN)*(TN*FP)*(TN*FN))
print(acc,f1,mcc)
#%%
# =============================================================================
# =============================================================================
# # GAN empty
# =============================================================================
# =============================================================================
total=3152
t_ev=2621
TP,TN,FP,FN=[2321-300,160,120,200+300]
acc=(TP+TN)/(TP+TN+FP+FN)
f1=(2*TP)/(2*TP+FP+FN)
mcc=((TP*TN)-(FP*FN))/np.sqrt((TP+FP)*(TP+FN)*(TN*FP)*(TN*FN))
print(acc,f1,mcc)
#%%
# =============================================================================
# 
# =============================================================================
# =============================================================================
# benchmark
# =============================================================================
total=3152
t_ev=2621
TP,TN,FP,FN=[450,460,90,1500]
acc=(TP+TN)/(TP+TN+FP+FN)
f1=(2*TP)/(2*TP+FP+FN)
mcc=((TP*TN)-(FP*FN))/np.sqrt((TP+FP)*(TP+FN)*(TN*FP)*(TN*FN))
print(acc,f1,mcc)
#%%%

# =============================================================================
# =============================================================================
# =============================================================================
# # # correlation plot for ivpq
# =============================================================================
# =============================================================================
# =============================================================================
corr={}


days=np.arange(3,18)
for d in days:
    cr=np.zeros((12,12))
    if d<10:
        
        filename='data/Armin_Data/July_0'+str(d)+'/pkl/J'+str(d)+'.pkl'
    else:
        filename='data/Armin_Data/July_'+str(d)+'/pkl/J'+str(d)+'.pkl'
    data=load_real_data(filename)
    for i in range(12):
        print(i)
        for j in range(12):
            if i >=j:
                cr[i,j]=np.corrcoef(data[i],data[j])[0,1]
                cr[j,i]=cr[i,j]
    
    sns.heatmap(cr)
    corr[d]=cr
#%%
for d in corr:
    print(d)
    sns.heatmap(corr[d])
    plt.show()
    
    
#%%

sns.heatmap(corr[15])
#%%
anom_select=[30855, 35292, 46381, 49019, 49998, 74174]
anom_select=[322691]

scale=1100
shift=1283000
for anom in anom_select:
        print(anom)
        anom=int(anom)

        plt.subplot(221)
        for i in [2]:
            plt.plot(select_1224[i][anom*int(SampleNum/2)-40*scale+shift:(anom*int(SampleNum/2)+40*scale+shift-20000)])
        plt.legend('A' 'B' 'C',fontsize= 20,loc=6)
        plt.yticks(fontsize=15)
        plt.ylim([7120,7200])
#        plt.figtext(.5,.9,'Temperature', fontsize=100, ha='center')
        plt.title('V (Volts)',fontsize= 30)
#        plt.xlabel('Timeslots',fontsize= 30)
#        plt.xticks(fontsize=15)
#        plt.yticks(fontsize=15)
#        
#            
        plt.subplot(222)
        for i in [3]:
            plt.plot(select_1224[i][anom*int(SampleNum/2)-40*scale+shift:(anom*int(SampleNum/2)+40*scale+shift-20000)])
#        plt.legend('A' 'B' 'C')
        plt.title('I (Amps)',fontsize= 30)  
        plt.yticks(fontsize=15)
        plt.ylim([100,150])

        
        plt.subplot(223)
        for i in [6]:
            plt.plot(select_1224[i][anom*int(SampleNum/2)-40*scale+shift:(anom*int(SampleNum/2)+40*scale+shift-20000)]/1000)
#        plt.legend('A' 'B' 'C') 
        plt.title('P (kW)',fontsize= 30)    
        plt.xlabel('Timeslots',fontsize= 30)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim([900,1040])
        
        plt.subplot(224)
        for i in [11]:
            plt.plot(select_1224[i][anom*int(SampleNum/2)-40*scale+shift:(anom*int(SampleNum/2)+40*scale+shift-20000)]/1000)
#        plt.legend('A' 'B' 'C')
        plt.title('Q (kVAR)',fontsize= 30)
        plt.xlabel('Timeslots',fontsize= 30)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

#        figname='figures/paper/huge_osc.pdf'
#        plt.savefig(figname)
        plt.show()#%%













