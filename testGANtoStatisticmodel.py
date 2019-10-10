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
# # train data prepreation
# =============================================================================
# =============================================================================
filename='data/Armin_Data/July_03/pkl/jul3.pkl'
def load_data(start,SampleNum,N,filename):
         #read a pickle file
    pkl_file = open(filename, 'rb')
    selected_data = pkl.load(pkl_file)
    pkl_file.close()
    for pmu in ['1224']:
        selected_data[pmu]=pd.DataFrame.from_dict(selected_data[pmu])
    features=['L1MAG','L2MAG', 'L3MAG','C1MAG',
       'C2MAG', 'C3MAG', 'PA', 'PB', 'PC', 'QA', 'QB', 'QC']
    
    print(selected_data.keys())
    select=[]
    for f in features:
        select.append(selected_data[pmu][f])
    
    
    selected_data=0
    select=np.array(select)
    
    print(select.shape)
    select=preprocessing.scale(select,axis=1)
    
#    selected_data=0
    end=start+SampleNum
    shift=int(SampleNum/2)
    
    train_data=np.zeros((N,12,SampleNum))
#    reduced_mean=np.zeros((12,20))
    for i in range(N):
        if i% 1000==0:
            print('iter num: %i', i)
        temp=select[:,start+i*shift:end+i*shift] 
        temp=(temp-temp.mean(axis=1).reshape(-1,1)) ## reduced mean
#        temp = preprocessing.scale(temp,axis=1)  ## standardized
#        reduced_mean=np.concatenate((reduced_mean,temp[:,0:20]),axis=1)
        train_data[i,:]=temp
    
    
    # convert shape of x_train from (60000, 28, 28) to (60000, 784) 
    # 784 columns per row
    
    return train_data#,select,selected_data#,select_proc,reduced_mean
#X_train=load_data()
#print(X_train.shape)
    
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
    
    generator.add(Dense(units=12*40))
    
    generator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return generator
g=create_generator()
g.summary()

#%%
def create_discriminator():
    discriminator=Sequential()
    discriminator.add(CuDNNLSTM(units=256,input_shape=(40,12),return_sequences=True))
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
    x = Reshape((40,12), input_shape=(12*40,1))(x)
    gan_output= discriminator(x)
    gan= Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan
gan = create_gan(d,g)
gan.summary()

#%%
batch_size=100
epochnum=1000

#%%
#%%

start,SampleNum,N=(0,40,1000)
#X_train = load_data(start,SampleNum,N)
filename=
X_train = load_data(start,SampleNum,N,filename)
batch_count = X_train.shape[0] / batch_size
#%%
X_train=X_train.reshape(N,12*SampleNum)
X_train=X_train.reshape(N,SampleNum,12)
#%%
generator= create_generator()
discriminator= create_discriminator()
gan = create_gan(discriminator, generator)

#%%

def training(generator,discriminator,gan,epochs, batch_size):
    
    scale=1
    for e in range(1,epochs+1 ):
        tik=time.clock()
        print("Epoch %d" %e)
        for _ in tqdm(range(batch_size)):
        #generate  random noise as an input  to  initialize the  generator
            noise= scale*np.random.normal(0,1, [batch_size, 100])
            noise=noise.reshape(batch_size,100,1)
            # Generate fake MNIST images from noised input
            generated_images = generator.predict(noise)
            generated_images = generated_images.reshape(batch_size,SampleNum,12)
#            print(generated_images.shape)
            # Get a random set of  real images
            image_batch =X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)]
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
        toc = time.clock()
        print(toc-tik)

#        if e == 1 or e % 5 == 0:
#           
            
#            plot_generated_images(e, generator)
#batch_size=0
tic = time.clock()   
training(generator,discriminator,gan,epochnum,batch_size)
toc = time.clock()

print(toc-tic)
#%%
##
#gan.save('GPU_gan_mul_LSTM_twolayer_N500000_e1000_b10_1225.h5')
#generator.save('GPU_generator_mul_LSTM_twolayer_N500000_e1000_b10_1225.h5')
#discriminator.save('GPU_discriminator_mul_LSTM_twolayer_N500000_e1000_b10_1225.h5')
#%%

gan=load_model('GPU_gan_mul_LSTM_twolayer_N500000_e1000_b100.h5')
generator=load_model('GPU_generator_mul_LSTM_twolayer_N500000_e1000_b100.h5')
discriminator=load_model('GPU_discriminator_mul_LSTM_twolayer_N500000_e1000_b100.h5')
#%%
filename='data/Armin_Data/July_13/pkl/J13.pkl'
start,SampleNum,N,filename=(0,40,500000,filename)

X_train= load_data(start,SampleNum,N,filename)
#batch_count = X_train.shape[0] / batch_size

#%%
X_train=X_train.reshape(N,12*SampleNum)
X_train=X_train.reshape(N,SampleNum,12)

#%%
rate=1000
shift=N/rate
scores=[]
for i in range(rate-1):
    temp=discriminator.predict_on_batch(X_train[int(i*shift):int((i+1)*shift)])
    scores.append(temp)
    print(i)

scores=np.array(scores)
scores=scores.ravel()
#%%

#%%

probability_mean=np.mean(scores)
a=scores-probability_mean

#%%
#fig_size = plt.rcParams["figure.figsize"]
# 
# 
## Set figure width to 12 and height to 9
#fig_size[0] = 8
#fig_size[1] = 6
#plt.plot(a.ravel())
#plt.show()
#%%
# =============================================================================
# =============================================================================
# # determining the higher and uper bound based on the train data
# =============================================================================
# =============================================================================
data = a

# Fit a normal distribution to the data:
mu, std = norm.fit(data)

# Plot the histogram.
plt.hist(data, bins=25, density=True, alpha=0.6, color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()

x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)


plt.show()
#%%
# =============================================================================
# =============================================================================
# #GAN model calling
# =============================================================================
# =============================================================================

gan=load_model('GPU_gan_mul_LSTM_twolayer_N500000_e1000_b100.h5')
generator=load_model('GPU_generator_mul_LSTM_twolayer_N500000_e1000_b100.h5')
discriminator=load_model('GPU_discriminator_mul_LSTM_twolayer_N500000_e1000_b100.h5')
# =============================================================================
# Reading the files in the data to make a for
# =============================================================================
files=os.listdir('data/Armin_Data')
#%%
selected_files=[]
for f in files:
    s=f.split('_')
    if 'July' in s:
        selected_files.append(f)
#%%
# =============================================================================
# make a place to save all 1224 events data wrt each day, whether my method or Alirezas
# =============================================================================
dst="figures/all_events"
os.mkdir(dst)
#%%
#for num,file in enumerate(selected_files):
for file in ['July_17']:
    num=14  
    if file == 'July_03':
        continue
# =============================================================================
#     extract train data for the selected day
# =============================================================================
    print(file)
    start,SampleNum,N=(0,40,500000)
    dir="data/Armin_Data/"+ file + "/pkl/"
#    selectedfile=os.listdir(dir+str(num+3))
    filename = dir+'J'+str(num+3)+'.pkl'
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
    zp=2
    
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
    
    dst="figures/all_events/"+file
#    os.mkdir(dst)
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
            plt.plot(select_1224[i][anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)])
        plt.legend('A' 'B' 'C')
        plt.title('V')
            
        plt.subplot(222)
        for i in [3,4,5]:
            plt.plot(select_1224[i][anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)])
        plt.legend('A' 'B' 'C')
        plt.title('I')  
        
        plt.subplot(223)
        for i in [6,7,8]:
            plt.plot(select_1224[i][anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)])
        plt.legend('A' 'B' 'C') 
        plt.title('P')    
        
        plt.subplot(224)
        for i in [9,10,11]:
            plt.plot(select_1224[i][anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)])
        plt.legend('A' 'B' 'C')
        plt.title('Q')    
        figname=dst+"/"+str(anom)
        plt.savefig(figname)
        plt.show()
    # =============================================================================
    # find the wide range of anomalies point to compare with Alirezas data    
    # =============================================================================
        low=anom*20-240
        high=anom*20+240
        rng=np.arange(low,high)
        event_points.append(rng)
    event_points=np.array(event_points).ravel()
    
    
#%%    
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
    dst="figures/all_events/"+file
    anomcsvfile=dst+"/common"+file+".csv"
    np.savetxt(anomcsvfile, common_anoms, delimiter=",")
    # =============================================================================
    # make folder to save Alirezas event in the same day
    # =============================================================================
    dst="figures/all_events/"+file
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
            plt.plot(select_1224[i][anom-240:(anom+240)])
        plt.legend('A' 'B' 'C')
        plt.title('V')
            
        plt.subplot(222)
        for i in [3,4,5]:
            plt.plot(select_1224[i][anom-240:(anom+240)])
        plt.legend('A' 'B' 'C')
        plt.title('I')  
        
        plt.subplot(223)
        for i in [6,7,8]:
            plt.plot(select_1224[i][anom-240:(anom+240)])
        plt.legend('A' 'B' 'C') 
        plt.title('P')    
        
        plt.subplot(224)
        for i in [9,10,11]:
            plt.plot(select_1224[i][anom-240:(anom+240)])
        plt.legend('A' 'B' 'C')
        plt.title('Q')    
        figname=dst+"/"+str(anom)
        plt.savefig(figname)
        plt.show()