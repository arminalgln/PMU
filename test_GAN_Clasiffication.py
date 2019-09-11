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
filename='data/Armin_Data/July_03/pkl/J3.pkl'
select_1224=load_real_data(filename)
#%%
#start,SampleNum,N=(0,40,500000)
#group={}
#group['0']=[]
#group['1']=[]
#for window in range(N):
#    if window>=0:
#        print(window)
#            
#        
#        plt.subplot(221)
#        for i in [0,1,2]:
#            plt.plot(select_1224[i][window*int(SampleNum/2):(window*int(SampleNum/2)+40)])
#        plt.legend('A' 'B' 'C')
#        plt.title('V')
#            
#        plt.subplot(222)
#        for i in [3,4,5]:
#            plt.plot(select_1224[i][window*int(SampleNum/2):(window*int(SampleNum/2)+40)])
#        plt.legend('A' 'B' 'C')
#        plt.title('I')  
#        
#        plt.subplot(223)
#        for i in [6,7,8]:
#            plt.plot(select_1224[i][window*int(SampleNum/2):(window*int(SampleNum/2)+40)])
#        plt.legend('A' 'B' 'C') 
#        plt.title('P')    
#        
#        plt.subplot(224)
#        for i in [9,10,11]:
#            plt.plot(select_1224[i][window*int(SampleNum/2):(window*int(SampleNum/2)+40)])
#        plt.legend('A' 'B' 'C')
#        plt.title('Q')    
#        plt.show()
#        
#        gr=input("which group?:   ")
#        
#        if not gr in group:
#            print('wrong')
#            gr=input("which group?:   ")
#            group[gr].append(window)
#        else:
#            group[gr].append(window)
#%%
#import _thread
#import threading
#start,SampleNum,N=(0,40,500000)
#eventwindow=[]
##group['0']=[]
##group['1']=[]
#thresh=427104
#while thresh<500000:
#    try:
#        for window in range(N):
#            if window>=thresh:
#                print(window)
#                    
#                
#                plt.subplot(221)
#                for i in [0,1,2]:
#                    plt.plot(select_1224[i][window*int(SampleNum/2):(window*int(SampleNum/2)+40)])
#                plt.legend('A' 'B' 'C')
#                plt.title('V')
#                    
#                plt.subplot(222)
#                for i in [3,4,5]:
#                    plt.plot(select_1224[i][window*int(SampleNum/2):(window*int(SampleNum/2)+40)])
#                plt.legend('A' 'B' 'C')
#                plt.title('I')  
#                
#                plt.subplot(223)
#                for i in [6,7,8]:
#                    plt.plot(select_1224[i][window*int(SampleNum/2):(window*int(SampleNum/2)+40)])
#                plt.legend('A' 'B' 'C') 
#                plt.title('P')    
#                
#                plt.subplot(224)
#                for i in [9,10,11]:
#                    plt.plot(select_1224[i][window*int(SampleNum/2):(window*int(SampleNum/2)+40)])
#                plt.legend('A' 'B' 'C')
#                plt.title('Q')    
#                plt.show()
##                time.sleep(1)
#    except KeyboardInterrupt:
#        window=input("which group?:   ")
#        
#        eventwindow.append(int(window))
#        thresh=int(window)
#                
#        real_event="data/Armin_Data/eventwindowbyhand.pkl"
#        output = open(real_event, 'wb')
#        pkl.dump(eventwindow, output)
#        output.close()
#%%
#pkl_file = open(real_event, 'rb')
#real_event = pkl.load(pkl_file)
#pkl_file.close()
#        
##%%
#import signal
#def interrupted(signum, frame):
#    print("Timeout!")
#signal.signal(signal.SIGALRM, interrupted)
#signal.alarm(5)
#try:
#    s = input("::>")
#except:
#    print("You are interrupted.")
#signal.alarm(0)
##%%
#        
#real_event="data/Armin_Data/categories.pkl"
#output = open(real_event, 'wb')
#pkl.dump(group, output)
#output.close()
##%%
#pkl_file = open(real_event, 'rb')
#real_event = pkl.load(pkl_file)
#pkl_file.close()
#    

#%%
# =============================================================================
# Reading the files in the data to make a for
# =============================================================================
files=os.listdir('figures/all_events/')
#%%
# =============================================================================
# =============================================================================
# =============================================================================
# # # take out anommalies# =============================================================================
# =============================================================================
# =============================================================================
anomalies={}
for num,file in enumerate(files):
    if num<13:
        if not file.endswith(".txt"):
            dir='figures/all_events/'
            dir=dir+file+"/GAN"
            tempfiles=os.listdir(dir)
            for f in tempfiles:
                if f.endswith(".csv"):
                    anomfile=dir+'/'+f
                    ta=pd.read_csv(anomfile)
                    anomalies[file]=ta.values
            print(dir)
            dir='figures/all_events/'
            dir=dir+file+"/GAN_voltage"
            tempfiles=os.listdir(dir)
            for f in tempfiles:
                if f.endswith(".csv"):
                    anomfile=dir+'/'+f
                    ta=pd.read_csv(anomfile)
                    anomalies[file+'v']=ta.values
            print(dir)
#%%
# =============================================================================
# =============================================================================
# # save all the animalies  for GAN model          
# =============================================================================
# =============================================================================
output = open('figures/all_events/All_GAN_anomalies.pkl', 'wb')
pkl.dump(anomalies, output)
output.close() 
        #%%
# =============================================================================
#         read anomalies
# =============================================================================
pkl_file = open('figures/all_events/All_GAN_anomalies.pkl', 'rb')
anomalies = pkl.load(pkl_file)
pkl_file.close()
#%%
select_1224=load_standardized_data('data/Armin_Data/July_03/pkl/J3.pkl')
#%%
start,SampleNum,N=(0,40,500000)
event_points={}
for anom in anomalies['July_03']:
    anom=int(anom)
    event_points[anom]=select_1224[0:12,anom*int(SampleNum/2)-120:(anom*int(SampleNum/2)+120)]
        
#%%
# =============================================================================
# =============================================================================
# #     calculate absolute of the events fft
# =============================================================================
# =============================================================================
fft_scores={}
fs=[]
for event in anomalies['July_03']:
    event=int(event)
    v=np.absolute(fft(event_points[event][0])[1:120])    
    i=np.absolute(fft(event_points[event][3])[1:120])
    p=np.absolute(fft(event_points[event][6])[1:120])    
    q=np.absolute(fft(event_points[event][9])[1:120])
    vi=np.concatenate((v,i))
    pq=np.concatenate((p,q))
    fft_scores[event]=np.concatenate((vi,pq))
    fs.append(np.concatenate((v,i)))
fs=np.array(fs)
    
#%%
# =============================================================================
# =============================================================================
# # classifying the events with fft
# =============================================================================
# =============================================================================
X=fs
mm=0
for n_clusters in np.arange(2,30):
    clusterer = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    if silhouette_avg >mm:
        mm=silhouette_avg
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
print(mm)
#%%
# =============================================================================
# =============================================================================
# # best so far
# =============================================================================
# =============================================================================
n_clusters=15
clusterer = KMeans(n_clusters=n_clusters, random_state=0)
cluster_labels = clusterer.fit_predict(X)

#%%
# =============================================================================
# =============================================================================
# # show some sample from each cluster in one day
# =============================================================================
# =============================================================================
    
#cl=0
for cl in range(n_clusters):
    count=0
    
    for num,event in enumerate(event_points):
        if cluster_labels[num]==cl:
            if count<20:
                print(cl)
                plt.subplot(221)
                for i in [0,1,2]:
                    plt.plot(event_points[event][i])
                plt.legend('A' 'B' 'C')
                plt.title('V')
                    
                plt.subplot(222)
                for i in [3,4,5]:
                    plt.plot(event_points[event][i])
                plt.legend('A' 'B' 'C')
                plt.title('I')  
                
                plt.subplot(223)
                for i in [6,7,8]:
                    plt.plot(event_points[event][i])
                plt.legend('A' 'B' 'C') 
                plt.title('P')    
                
                plt.subplot(224)
                for i in [9,10,11]:
                    plt.plot(event_points[event][i])
                plt.legend('A' 'B' 'C')
                plt.title('Q')    
                plt.show()
    
                count+=1
#    count+=1
    #%%
for event in anomalies['July_03']:
    print(event)
    event=int(event)
    plt.subplot(221)
    for i in [0,1,2]:
        plt.plot(event_points[event][i])
    plt.legend('A' 'B' 'C')
    plt.title('V')
        
    plt.subplot(222)
    for i in [3,4,5]:
        plt.plot(event_points[event][i])
    plt.legend('A' 'B' 'C')
    plt.title('I')  
    
    plt.subplot(223)
    for i in [6,7,8]:
        plt.plot(event_points[event][i])
    plt.legend('A' 'B' 'C') 
    plt.title('P')    
    
    plt.subplot(224)
    for i in [9,10,11]:
        plt.plot(event_points[event][i])
    plt.legend('A' 'B' 'C')
    plt.title('Q')    
    plt.show()
    
    v=fft(event_points[event][0]-np.mean(event_points[event][0]))
    plt.plot(v[1:120])
    plt.show()
    
    
    i=fft(event_points[event][3]-np.mean(event_points[event][3]))
    plt.plot(i[1:120])
    plt.show()
     #%%

        
    
#%%
#data_files=os.listdir('data/Armin_Data')
#event_points={}
#start,SampleNum,N=(0,40,500000)
#for day in anomalies:
#    print(day)
#    anoms=anomalies[day]
#    dir="data/Armin_Data/"+ day + "/pkl/"
#    selectedfile=os.listdir(dir)[0]
#    filename = dir + selectedfile
#    select_1224=load_standardized_data(filename)
#    event_points[day]={}
#    for anom in anoms:
#        anom=int(anom)
#        event_points[day][anom]=select_1224[0:12,anom*int(SampleNum/2)-120:(anom*int(SampleNum/2)+120)]
#        
#        #%%
#        
#eventpointsfile="data/Armin_Data/event_hand_standardized.pkl"
##%%
#output = open(eventpointsfile, 'wb')
#pkl.dump(event_points, output)
#output.close()
##%%
#pkl_file = open(eventpointsfile, 'rb')
#event_points = pkl.load(pkl_file)
#pkl_file.close()
#%%
        
# =============================================================================
# =============================================================================
# #         classifying first day events by hand
# =============================================================================
# =============================================================================
#group={}
##%%
#for event in event_points['July_03']:
#    if event>0:
#        print(event)
#            
#        plt.subplot(221)
#        for i in [0,1,2]:
#            plt.plot(event_points['July_03'][event][i])
#        plt.legend('A' 'B' 'C')
#        plt.title('V')
#            
#        plt.subplot(222)
#        for i in [3,4,5]:
#            plt.plot(event_points['July_03'][event][i])
#        plt.legend('A' 'B' 'C')
#        plt.title('I')  
#        
#        plt.subplot(223)
#        for i in [6,7,8]:
#            plt.plot(event_points['July_03'][event][i])
#        plt.legend('A' 'B' 'C') 
#        plt.title('P')    
#        
#        plt.subplot(224)
#        for i in [9,10,11]:
#            plt.plot(event_points['July_03'][event][i])
#        plt.legend('A' 'B' 'C')
#        plt.title('Q')    
#        plt.show()
#        
#        gr=input("which group?:   ")
#        
#        if not gr in group:
#            permission=input('sure?')
#            if permission=='y':
#                group[gr]=[event]
#            else:
#                gr=input("which group?:   ")
#                if not gr in group:
#                    permission=input('sure?')
#                    if permission=='y':
#                        group[gr]=[event]
#        else:
#            group[gr].append(event)
##%%
## =============================================================================
## =============================================================================
## # save the groups        
## =============================================================================
## =============================================================================
#
#categoriesfile="data/Armin_Data/categories.pkl"
##%%
#output = open(categoriesfile, 'wb')
#pkl.dump(group, output)
#output.close()
##%%
#pkl_file = open(categoriesfile, 'rb')
#saved_group = pkl.load(pkl_file)
#pkl_file.close()
    
#%%
# =============================================================================
# =============================================================================
# # show the mean value of each category
# =============================================================================
## =============================================================================
#
#count=0
#for g in saved_group:
#    group_size=len(saved_group[g])
#    for event in saved_group[g]:
#        if count==0:
#            mean_events=event_points['July_03'][event]
#            count=1
#        else:
#            mean_events+=event_points['July_03'][event]
#    mean_events=mean_events/group_size
#    print("group name: ",g,"  number of events:  ",group_size)
#    plt.subplot(221)
#    for i in [0,1,2]:
#        plt.plot(mean_events[i])
#    plt.legend('A' 'B' 'C')
#    plt.title('V')
#        
#    plt.subplot(222)
#    for i in [3,4,5]:
#        plt.plot(mean_events[i])
#    plt.legend('A' 'B' 'C')
#    plt.title('I')  
#    
#    plt.subplot(223)
#    for i in [6,7,8]:
#        plt.plot(mean_events[i])
#    plt.legend('A' 'B' 'C') 
#    plt.title('P')    
#    
#    plt.subplot(224)
#    for i in [9,10,11]:
#        plt.plot(mean_events[i])
#    plt.legend('A' 'B' 'C')
#    plt.title('Q')    
#    plt.show()
#    print(".......................")






#%%
# =============================================================================
# =============================================================================
# #         save the anomalies standardized data for 15 days
# =============================================================================
# =============================================================================
#anomcsvfile="data/Armin_Data/anomsknnformat.pkl"
#output = open(anomcsvfile, 'wb')
#pkl.dump(event_points, output)
#output.close()
#%%
# =============================================================================
# =============================================================================
# # read event_points
# =============================================================================
## =============================================================================
#anomcsvfile="data/Armin_Data/anomsknnformat.pkl"
#pkl_file = open(anomcsvfile, 'rb')
#event_points = pkl.load(pkl_file)
#pkl_file.close()
##%%
#X=[]
#for day in event_points:
#    for event in event_points[day]:
#        X.append(event_points[day][event].ravel())
#X=np.array(X)
##%%
#kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
##%%
#
#for n_clusters in np.arange(10,40):
#    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
#    cluster_labels = clusterer.fit_predict(X)
#    silhouette_avg = silhouette_score(X, cluster_labels)
#    print("For n_clusters =", n_clusters,
#          "The average silhouette_score is :", silhouette_avg)
#
##%%
##pkl_file = open(anomcsvfile, 'rb')
##test = pkl.load(pkl_file)
#pkl_file.close()
##%%
#similarity_matrix=[]
#similarity_scores={}
#tik=time.clock()
#for day1 in event_points:
#    similarity_scores[day1]={}
#    print(day1)
#    for anom1 in event_points[day1]:
#        print(anom1)
#        temp_similarity=[]
#        
#        similarity_scores[day1][anom1]={}
#        
#        x1=event_points[day1][anom1][::3]-np.mean(event_points[day1][anom1][::3],axis=1).reshape(4,1)
#        x1=x1.ravel()
#        
#        for day2 in event_points:
#            print(day2)
#            similarity_scores[day1][anom1][day2]={}
#            
#            for anom2 in event_points[day2]:
#                print(anom2)
#                x2=event_points[day2][anom2][::3]-np.mean(event_points[day2][anom2][::3],axis=1).reshape(4,1)
#                x2=x2.ravel()
#
##        plt.plot(event_points['July_10'][i][0]-np.mean(event_points['July_10'][i][0]))
##        plt.plot(event_points['July_10'][j][0]-np.mean(event_points['July_10'][j][0]))
##        plt.show()
#                d, path = fastdtw(x1, x2, dist=euclidean_norm)
#                print(d)
#                similarity_scores[day1][anom1][day2][anom2]=d
#                temp_similarity.append(d)
#                
#        temp_similarity=np.array(temp_similarity)
#        similarity_matrix.append(temp_similarity)
#similarity_matrix=np.array(similarity_matrix)
#toc = time.clock()
#print(toc-tik)
#time_4features=toc-tik
#        print(d)
#        plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
#        plt.plot(path[0], path[1], 'w')
#        plt.show()
##        print('...........................................................')
##%%
## =============================================================================
## =============================================================================
## # calculating fft for each event and save them
## =============================================================================
## =============================================================================
#
#
#fft_scores={}
#total_events=0
#all_evnets_scores=[]
#for day1 in event_points:
#    fft_scores[day1]={}
##    print(day1)
#
#    for count,anom1 in enumerate(event_points[day1]):
##        print(anom1)
#        total_events+=1
#        x1=event_points[day1][anom1][::3]-np.mean(event_points[day1][anom1][::3],axis=1).reshape(4,1)      
#        
#        fft_scores[day1][anom1]=np.concatenate((np.fft.fftn(x1)[:,0:120].real.ravel(),np.fft.fftn(x1)[:,0:120].imag.ravel()),axis=None)
#   # =============================================================================
## =============================================================================
## # make trainig data with fft output
## =============================================================================
## =============================================================================
#
#
#        all_evnets_scores.append(np.concatenate((np.fft.fftn(x1)[:,0:120].real.ravel(),np.fft.fftn(x1)[:,0:120].imag.ravel()),axis=None))
#        
#        if count% 500==0:
#            print('iter num: %count', count)
#print(total_events)
#anomcsvfile="data/Armin_Data/fftscores.pkl"
#output = open(anomcsvfile, 'wb')
#pkl.dump(fft_scores, output)
#output.close()
#
#
#all_evnets_scores=np.array(all_evnets_scores)
#
#all_evnets_scores_file="data/Armin_Data/all_evnets_scores_file.pkl"
#output = open(all_evnets_scores_file, 'wb')
#pkl.dump(all_evnets_scores, output)
#output.close()
#
##%%
#X=all_evnets_scores
#
#for n_clusters in np.arange(10,50):
#    clusterer = KMeans(n_clusters=n_clusters, random_state=0)
#    cluster_labels = clusterer.fit_predict(X)
#    silhouette_avg = silhouette_score(X, cluster_labels)
#    print("For n_clusters =", n_clusters,
#          "The average silhouette_score is :", silhouette_avg)
##%%
## =============================================================================
## =============================================================================
## # best cluster number by fft is 18 based in silhouette
## =============================================================================
## =============================================================================
#n_clusters=18
#clusterer = KMeans(n_clusters=n_clusters, random_state=0)
#cluster_labels = clusterer.fit_predict(X)
##%%
## =============================================================================
## =============================================================================
## # predict the labels for main dataset
## =============================================================================
## =============================================================================
#labels={}
#start=0
#end=0
#for day in event_points:
#    num_anom=len(event_points[day].keys())
#    end=start+num_anom
#    selected_fft=all_evnets_scores[start:end]
#    labels[day]=clusterer.fit_predict(selected_fft)
#    start=end
#    print(day)
##%%
## =============================================================================
## =============================================================================
## # show some sample from each cluster in one day
## =============================================================================
## =============================================================================
#    
#count=0
#for anom in event_points['July_03']:
#    print(labels['July_03'][count])
#    plt.subplot(121)
#    plt.plot(event_points['July_03'][anom][0])
#    plt.subplot(122)
#    plt.plot(event_points['July_03'][anom][3])
#    plt.show()
#    count+=1
#    
##%%
#
##%%%
#for day1 in ['July_03']:
#    similarity_scores[day1]={}
#    print(day1)
#    for anom1 in event_points[day1]:
#        temp_similarity=[]
#        print(anom1)
#        similarity_scores[day1][anom1]={}
#        
#        x1=event_points[day1][anom1][::3]-np.mean(event_points[day1][anom1][::3],axis=1).reshape(4,1)
#        x1=x1[3]
#        ff=np.fft.fft(x1)
#        freq = np.fft.fftfreq(x1.shape[-1])
#        
#        widths = np.arange(1, 240)
#        cwtmatr = signal.cwt(x1, signal.ricker,widths)
#        plt.subplot(131)
#        plt.plot(freq, ff.real, freq, ff.imag)
#        plt.subplot(132)
#        plt.plot(x1)
#        plt.subplot(133)
#        plt.imshow(cwtmatr, extent=[-1, 1, 31, 1], cmap='PRGn', aspect='auto',
#              vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
#        plt.show()
        #%%
