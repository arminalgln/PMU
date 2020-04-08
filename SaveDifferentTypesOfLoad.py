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
import scipy
import loading_data
from loading_data import load_real_data, load_standardized_data,load_train_data,load_train_data_V,load_train_vitheta_data_V,load_data_with_features,load_standardized_data_with_features

#%% 
#%%
# =============================================================================
# =============================================================================
# # read one file of the PMU data , each file is for 10 minutes 
# =============================================================================
# =============================================================================
#%%
# importing data from a file function
def OneFileImport(filename,dir):
    dir_name=dir
    base_filename=filename
    path=os.path.join(dir_name, base_filename)
    imported_data=pd.read_csv(path)
    return imported_data
    
#%%
# =============================================================================
# =============================================================================
# #     save data with V I and theta
# =============================================================================
# =============================================================================
for n in [3]:
    if n<10:
        dir="../../UCR/PMU data/Data/July_0"+str(n)+"/"
    else:
        dir="../../UCR/PMU data/Data/July_"+str(n)+"/"
#dir='data/Armin_Data/July_03'
#os.listdir('../../UCR/PMU data/Data')
    foldernames=os.listdir(dir)
    selected_files=np.array([])
    for f in foldernames:
        spl=f.split('_')
        if 'Hunter' in spl:
            selected_files=np.append(selected_files,f)
    selected_files
    filenames1224=natsorted(selected_files)
    filenames1224
    def OneFileImport(filename,dir):
        dir_name=dir
        base_filename=filename
        path=os.path.join(dir_name, base_filename)
        imported_data=pd.read_csv(path)
        return imported_data
    whole_data=np.array([])
    for count,file in enumerate(filenames1224):
        print(count,file)
        cosin={}
    #    Reacive={}
    #    keys={}
    #    pf={}
        
        selected_data=OneFileImport(file,dir)    
        
        cosin['TA']=np.cos((selected_data['L1Ang']-selected_data['C1Ang'])*(np.pi/180))
        cosin['TB']=np.cos((selected_data['L2Ang']-selected_data['C2Ang'])*(np.pi/180))
        cosin['TC']=np.cos((selected_data['L3Ang']-selected_data['C3Ang'])*(np.pi/180))
            
    #    Reacive['A']=selected_data['L1Mag']*selected_data['C1Mag']*(np.sin((selected_data['L1Ang']-selected_data['C1Ang'])*(np.pi/180)))
    #    Reacive['B']=selected_data['L2Mag']*selected_data['C2Mag']*(np.sin((selected_data['L2Ang']-selected_data['C2Ang'])*(np.pi/180)))
    #    Reacive['C']=selected_data['L3Mag']*selected_data['C3Mag']*(np.sin((selected_data['L3Ang']-selected_data['C3Ang'])*(np.pi/180)))
        #   
        #pf['A']=Active['A']/np.sqrt(np.square(Active['A'])+np.square(Reacive['A']))
        #pf['B']=Active['B']/np.sqrt(np.square(Active['B'])+np.square(Reacive['B']))
        #pf['C']=Active['C']/np.sqrt(np.square(Active['C'])+np.square(Reacive['C']))
        
        
        selected_data['TA']=cosin['TA']
        selected_data['TB']=cosin['TB']
        selected_data['TC']=cosin['TC']
        
        selected_data=selected_data.drop(columns=['Unnamed: 0','L1Ang','L2Ang','L3Ang','C1Ang','C2Ang','C3Ang'])
    
    #    
    #    selected_data['QA']=Reacive['A']
    #    selected_data['QB']=Reacive['B']
    #    selected_data['QC']=Reacive['C'] 
    #    
        if count==0:
            whole_data=selected_data.values
        else:
            whole_data=np.append(whole_data,selected_data.values,axis=0)
#            
#    k=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG','L1Ang','L2Ang','L3Ang','C1Ang','C2Ang','C3Ang']
    k=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG','TA', 'TB', 'TC']
    day_data={}
    day_data['1224']={}
    c=0
    for key in k:
        day_data['1224'][key]=whole_data[:,c]
        c+=1
        
#    if n<10:
#        dir="data/Armin_Data/July_sep_0"+str(n)+"/pkl"
#    else:
#        dir="data/Armin_Data/July_sep_"+str(n)+"/pkl"
#    dir_name=dir
#    os.mkdir(dir_name)
        # write python dict to a file
    if n<10:
        dir="data/Armin_Data/July_0"+str(n)+"/pkl/rawdata" + str(n) + ".pkl"
    else:
        dir="data/Armin_Data/July_"+str(n)+"/pkl/rawdata" + str(n) + ".pkl"
    output = open(dir, 'wb')
    pkl.dump(day_data, output)
    output.close()
    print(n)
    
    #%%
filename='data/Armin_Data/July_03/pkl/rawdata3.pkl'
k=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG','TA', 'TB', 'TC']
#dds14=load_standardized_data_with_features(filename,k)
dd3=load_data_with_features(filename,k)
start,SampleNum,N=(0,40,500000)
#filename='data/Armin_Data/July_03/pkl/julseppf3.pkl'
#k=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG','TA', 'TB', 'TC']
#tt14=load_train_vitheta_data_V(start,SampleNum,N,filename,k)
#%%
%matplotlib inline
ev=[53766,355644]
dst='clusters/vit/111111111/cap'
show(ev,dd3,dst)
%matplotlib auto

#%%

def show(events,select_1224,dst):
    SampleNum=40
    for anom in events:
            print(anom)
            anom=int(anom)
#            anom=events[anom]
#            print(anom)
            
            plt.subplot(221)
            for i in [0,1,2]:
                plt.plot(select_1224[i][anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)])
            plt.legend('A' 'B' 'C')
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
            figname=dst+"/"+str(anom)
            plt.savefig(figname)
            plt.title('T')    
             
            plt.show()
#%%


def just_show(events,select_1224):
    shift=240
    SampleNum=40
    for anom in events:
            print(anom)
            anom=int(anom)
#            anom=events[anom]
#            print(anom)
            
            plt.subplot(221)
            for i in [0,1,2]:
                plt.plot(select_1224[i][anom*int(SampleNum/2)-shift:(anom*int(SampleNum/2)+shift)])
#            plt.legend('A' 'B' 'C')
            plt.title('V')
                
            plt.subplot(222)
            for i in [3,4,5]:
                plt.plot(select_1224[i][anom*int(SampleNum/2)-shift:(anom*int(SampleNum/2)+shift)])
#            plt.legend('A' 'B' 'C')
            plt.title('I')  
            
            plt.subplot(223)
            for i in [6,7,8]:
                
                plt.plot(select_1224[i][anom*int(SampleNum/2)-shift:(anom*int(SampleNum/2)+shift)])
#            plt.legend('A' 'B' 'C') 
#            figname=dst+"/"+str(anom)
#            plt.savefig(figname)
            plt.title('T')    
             
            plt.show()
            
        #%%
x = data_matlab[2]
w = np.fft.fft(x)
freqs = np.fft.fftfreq(len(x))

for coef,freq in zip(w,freqs):
    if coef:
        print('{c:>6} * exp(2 pi i t * {f})'.format(c=coef,f=freq))


#%%
v=0
for inx,f in enumerate(w):
    if inx>0:
        if np.absolute(f)>v:
            v=np.absolute(np.real(f))
            bid=inx
        
print(freqs[bid])
