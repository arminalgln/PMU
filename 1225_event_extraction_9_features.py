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

import loading_data
from loading_data import load_train_vitheta_data_1225,load_real_data, load_standardized_data,load_train_data,load_train_data_V,load_train_vitheta_data_V,load_data_with_features,load_standardized_data_with_features


#%%
   
#%%
# =============================================================================
# =============================================================================
# #     save data with V I and theta for 1225
# =============================================================================
# =============================================================================
filename='Raw_data/1225/data'
#os.listdir(filename)
#
pkl_file = open(filename, 'rb')
selected_data = pkl.load(pkl_file)
pkl_file.close()
cosin={}
#    Reacive={}
#    keys={}
#    pf={}

    
cosin['TA']=np.cos((selected_data['L1ANG']-selected_data['C1ANG'])*(np.pi/180))
cosin['TB']=np.cos((selected_data['L2ANG']-selected_data['C2ANG'])*(np.pi/180))
cosin['TC']=np.cos((selected_data['L3ANG']-selected_data['C3ANG'])*(np.pi/180))
    
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
        
k=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG','TA', 'TB', 'TC']
day_data={}
for key in k:
    day_data[key]=selected_data[key]
        

dir='Raw_data/1225/VIT.pkl'
output = open(dir, 'wb')
pkl.dump(day_data, output)
output.close()

#%%


# =============================================================================
# =============================================================================
# # train data prepreation
# =============================================================================
# =============================================================================
#start,SampleNum,N=(0,40,500000)
#filename='Raw_data/1225/VIT.pkl'
#k=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG','TA', 'TB', 'TC']
##%%
#dds=load_standardized_data_with_features(filename,k)
##%%
#dd=load_data_with_features(filename,k)
#%%
# =============================================================================
# =============================================================================
# # real data for 1225 VIT
# =============================================================================
# =============================================================================
filename='Raw_data/1225/VIT.pkl'
pkl_file = open(filename, 'rb')
selected_data_1225_normal = pkl.load(pkl_file)
pkl_file.close()
#%%
# =============================================================================
# =============================================================================
# # data without key
# =============================================================================
# =============================================================================
selected_data_1225=[]
for f in k:
    selected_data_1225.append(selected_data_1225_normal[f])
#%%
start,SampleNum,N=(0,40,500000)
filename='Raw_data/1225/VIT.pkl'
k=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG','TA', 'TB', 'TC']
tt=load_train_vitheta_data_1225(start,SampleNum,N,filename,k)
#%%
X_train = tt
scores={}
probability_mean={}
anomalies={}
kkk=k[0:1]
for idx,key in enumerate(kkk):
    print(key)
    X_train_temp=X_train[:,idx]
#X_train.reshape(N,3*SampleNum)
    X_train_temp=X_train_temp.reshape(N,SampleNum,1)

    id=int(np.floor(idx/3))
    mode=k[id*3]
#    dis_name='dis_sep_onelearn_'+mode+'.h5'
#    print(dis_name)
#    
#    discriminator=load_model(dis_name)
    
    
    rate=1000
    shift=N/rate
    scores[key]=[]
    for i in range(rate-1):
        temp=discriminator.predict_on_batch(X_train_temp[int(i*shift):int((i+1)*shift)])
        scores[key].append(temp)
#        print(i)
    
    scores[key]=np.array(scores[key])
    scores[key]=scores[key].ravel()
    
    probability_mean[key]=np.mean(scores[key])
    data=scores[key]-probability_mean[key]
    
    mu, std = norm.fit(data)
    
    zp=3
    
    high=mu+zp*std
    low=mu-zp*std
    
    anomalies[key]=np.union1d(np.where(data>=high)[0], np.where(data<=low)[0])
    print(anomalies[key].shape)
    
#%%
# =============================================================================
# =============================================================================
# # plot 1225
# =============================================================================
# =============================================================================

def show_1225(events):
    SampleNum=40
    for anom in events:
            anom=int(anom)
            print(anom)
            
            plt.subplot(221)
            for i in [0,1,2]:
                plt.plot(selected_data_1225[i][anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)])
            plt.legend('A' 'B' 'C')
            plt.title('V')
                
            plt.subplot(222)
            for i in [3,4,5]:
                plt.plot(selected_data_1225[i][anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)])
            plt.legend('A' 'B' 'C')
            plt.title('I')  
            
            plt.subplot(223)
            for i in [6,7,8]:
                plt.plot(selected_data_1225[i][anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)])
            plt.legend('A' 'B' 'C') 
            plt.title('T')      
            plt.show() 
#%%
X_train = tt
            #%%
def adam_optimizer():
    return adam(lr=0.0002, beta_1=0.5)
#%%
def create_generator():
    generator=Sequential()
    generator.add(CuDNNLSTM(units=256,input_shape=(100,1),return_sequences=True))
    generator.add(LeakyReLU(0.2))
    
    generator.add(CuDNNLSTM(units=512))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=512))
    generator.add(LeakyReLU(0.2))
#    
#    generator.add(LSTM(units=1024))
#    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=1*40))
    
    generator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return generator
g=create_generator()
g.summary()

#%%
def create_discriminator():
    discriminator=Sequential()
    discriminator.add(CuDNNLSTM(units=256,input_shape=(40,1),return_sequences=True))
    discriminator.add(LeakyReLU(0.2))
#    discriminator.add(Dropout(0.3))
    discriminator.add(CuDNNLSTM(units=512))
    discriminator.add(LeakyReLU(0.2))
#    
    discriminator.add(Dense(units=512))
    discriminator.add(LeakyReLU(0.2))
#    discriminator.add(Dropout(0.3))
#       
#    discriminator.add(LSTM(units=256))
#    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Dense(units=1, activation='sigmoid'))
    
    discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return discriminator
d =create_discriminator()
d.summary()
#%%
def create_gan(discriminator, generator):
    discriminator.trainable=False
    gan_input = Input(shape=(100,1))
    x = generator(gan_input)
    x = Reshape((40,1), input_shape=(1*40,1))(x)
    gan_output= discriminator(x)
    gan= Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan
gan = create_gan(d,g)
gan.summary()

#%%
batch_size=5
epochnum=2


#%%

start,SampleNum,N=(0,40,500000)
#X_train = load_data(start,SampleNum,N)
#filename=
X_train = tt
batch_count = X_train.shape[0] / batch_size
##%%
#X_train=X_train.reshape(N,3*SampleNum)
#X_train=X_train.reshape(N,SampleNum,3)
#%%
rnd={}
for i in range(epochnum):
    rnd[i]=np.random.randint(low=0,high=N,size=batch_size)
#    show(rnd[i])
        

#%%
generator= create_generator()
discriminator= create_discriminator()
gan = create_gan(discriminator, generator)

#%%
all_scores=[]
def training(generator,discriminator,gan,epochs, batch_size,all_scores):
#    all_scores=[]
    scale=1
    for e in range(1,epochs+1 ):
        all_score_temp=[]
        tik=time.clock()
        print("Epoch %d" %e)
        for _ in tqdm(range(batch_size)):
        #generate  random noise as an input  to  initialize the  generator
            noise= scale*np.random.normal(0,1, [batch_size, 100])
            noise=noise.reshape(batch_size,100,1)
            # Generate fake MNIST images from noised input
            generated_images = generator.predict(noise)
            generated_images = generated_images.reshape(batch_size,SampleNum,1)
#            print(generated_images.shape)
            # Get a random set of  real images
#            random.seed(0)
            image_batch =X_train_temp[rnd[e-1]]
#            print(image_batch.shape)
            #Construct different batches of  real and fake data 
            X= np.concatenate([image_batch, generated_images])
            
            # Labels for generated and real data
            y_dis=np.zeros(2*batch_size)
            y_dis[:batch_size]=0.9
            
            #Pre train discriminator on  fake and real data  before starting the gan. 
            discriminator.trainable=True
            discriminator.train_on_batch(X, y_dis)
            
            #Tricking the noised input of the Generator as real data
            noise= scale*np.random.normal(0,1, [batch_size, 100])
            noise=noise.reshape(batch_size,100,1)
            y_gen = np.ones(batch_size)
            
            # During the training of gan, 
            # the weights of discriminator should be fixed. 
            #We can enforce that by setting the trainable flag
            discriminator.trainable=False
            
            #training  the GAN by alternating the training of the Discriminator 
            #and training the chained GAN model with Discriminator’s weights freezed.
            gan.train_on_batch(noise, y_gen)
            
            rate=1000
            shift=N/rate
            all_score_temp=[]
            for i in range(rate-1):
                temp=discriminator.predict_on_batch(X_train_temp[int(i*shift):int((i+1)*shift)])
                all_score_temp.append(temp)
    #                print(i)
            all_score_temp=np.array(all_score_temp)
            all_score_temp=all_score_temp.ravel()
            all_scores.append(all_score_temp)
        toc = time.clock()
        print(toc-tik)
            

#%%
kk=['L1MAG']
for idx,key in enumerate(kk):
    X_train_temp=X_train[:,(idx)]
#X_train.reshape(N,3*SampleNum)
    X_train_temp=X_train_temp.reshape(N,SampleNum,1)
    tic = time.clock()   
    training(generator,discriminator,gan,epochnum,batch_size,all_scores)
    toc = time.clock()
    print(toc-tic)
#    
#    gan_name='gan_sep_onelearn_good_09_'+key+'.h5'
#    gen_name='gen_sep_onelearn_good_09_'+key+'.h5'
#    dis_name='dis_sep_onelearn_good_09_'+key+'.h5'
#    print(dis_name)
#    gan.save(gan_name)
#    generator.save(gen_name)
#    discriminator.save(dis_name)

    
