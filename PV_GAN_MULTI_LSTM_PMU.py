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

import pickle
import operator
import math
from sklearn import preprocessing
from keras.models import load_model
import time
from scipy.stats import norm

#%% 
def load_data(start,SampleNum,N):
         #read a pickle file
    dir_name="data/sorted"
    filename=os.listdir(dir_name)
    path=os.path.join(dir_name,filename[3])
    pkl_file = open(path, 'rb')
    selected_data = pickle.load(pkl_file)
    pkl_file.close()
    features=['L1MAG','L2MAG', 'L3MAG','C1MAG',
       'C2MAG', 'C3MAG', 'PA', 'PB', 'PC', 'QA', 'QB', 'QC']
    
    select=[]
    for f in features:
        select.append(selected_data[f].iloc[0:int(N*SampleNum/2)+20].values)
    
    select=np.array(select)
    
#    select=preprocessing.scale(select,axis=1)
    
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
    
    return train_data,select,selected_data
#,select_proc,reduced_mean
#X_train=load_data()
#print(X_train.shape)
#%%
def load_data_manyfiles(start,SampleNum,N):
         #read a pickle file
    dir_name="data/sorted"
    filename=os.listdir(dir_name)
    
    fnum=0
    for file in filename:
        
        path=os.path.join(dir_name,file)
        pkl_file = open(path, 'rb')
        selected_data = pickle.load(pkl_file)
        pkl_file.close()
        features=['L1MAG','L2MAG', 'L3MAG','C1MAG',
           'C2MAG', 'C3MAG', 'PA', 'PB', 'PC', 'QA', 'QB', 'QC']
        
        select=[]
        for f in features:
            select.append(selected_data[f].iloc[0:int(N*SampleNum/2)+20].values)
        
        select=np.array(select)
        
    #    select=preprocessing.scale(select,axis=1)
        
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

        if fnum==0:
            whole_selected=select
            whole_train=train_data
        else:
            whole_selected=np.concatenate((whole_selected,select),axis=1)
            whole_train=np.concatenate((whole_train,train_data),axis=0)
        
        fnum+=1
    return whole_train,whole_selected
#,select_proc,reduced_mean
#X_train=load_data()
#print(X_train.shape)
    #%%
start,SampleNum,N=(0,40,2000)
X_train, selected = load_data_manyfiles(start,SampleNum,N)
print(X_train.shape,selected.shape)


#%%
def adam_optimizer():
    return adam(lr=0.0002, beta_1=0.5)
#%%
def create_generator():
    generator=Sequential()
    generator.add(CuDNNLSTM(units=256,input_shape=(100,1)))
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
    discriminator.add(CuDNNLSTM(units=256,input_shape=(40,12)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
#    
    discriminator.add(Dense(units=512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
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
def plot_generated_images(epoch, generator, examples=100, dim=(10,10), figsize=(10,10)):
    scale=1
    noise= scale*np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100,40,1)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.plot(generated_images[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image %d.png' %epoch)
    return generated_images
    
#%%
batch_size=100
epochnum=10

#%%

start,SampleNum,N=(0,40,2000)
X_train, selected = load_data(start,SampleNum,N)
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

gan.save('PV_GPU_gan_mul_LSTM_N2000_e100_b100.h5')
generator.save('PV_GPU_generator_mul_LSTM_N2000_e100_b100.h5')
discriminator.save('PV_GPU_discriminator_mul_LSTM_N2000_e100_b100.h5')
#%%

gan=load_model('PV_GPU_gan_mul_LSTM_N2000_e100_b100.h5')
generator=load_model('PV_GPU_generator_mul_LSTM_N2000_e100_b100.h5')
discriminator=load_model('PV_GPU_discriminator_mul_LSTM_N2000_e100_b100.h5')
#%%

start,SampleNum,N=(0,40,2000)
X_train, selected,selected_data = load_data(start,SampleNum,N)
batch_count = X_train.shape[0] / batch_size

#%%
X_train=X_train.reshape(N,12*SampleNum)
X_train=X_train.reshape(N,SampleNum,12)
#%%
a=discriminator.predict_on_batch(X_train)
#%%
probability_mean=np.mean(a)
a=a-probability_mean

#%%
fig_size = plt.rcParams["figure.figsize"]
 
 
# Set figure width to 12 and height to 9
fig_size[0] = 8
fig_size[1] = 6
plt.plot(a.ravel())
plt.ylabel('Event score')
plt.xlabel('training sample number')
#plt.ylim([.85,.95])
plt.savefig('probability score')
plt.show()
#%%

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

plt.savefig('normalpdfscore')
plt.show()
#%%
stdnum=3
high=mu+stdnum*std
low=mu-stdnum*std

fig_size = plt.rcParams["figure.figsize"]
 
 
# Set figure width to 12 and height to 9
fig_size[0] = 8
fig_size[1] = 6
anoms=np.union1d(np.where(a>=high)[0], np.where(a<=low)[0])
print(np.union1d(np.where(a>=high)[0], np.where(a<=low)[0]).shape)
tt=X_train.reshape(N,12*SampleNum)
tt=X_train.reshape(N,12,SampleNum)
#%%

normal=np.arange(100,110)
for i in anoms :
    print(i*int(SampleNum/2))
    for j in range(12):
        plt.plot(tt[i][j])
    plt.legend(('vol', 'curr', 'p','q'),shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=16)
    plt.show()
    

#%%
selected=pd.DataFrame(selected)
selected=selected.T

#%%
fig_size = plt.rcParams["figure.figsize"]
 
 
# Set figure width to 12 and height to 9
fig_size[0] = 10
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
start=0
dur=N*20
end=start+dur
selected['color']='b'
for i in anoms:
#    print(i)
    selected['color'].iloc[i*int(SampleNum/2):((i+1)*int(SampleNum/2)+40)]='r'

markers_on=np.where(selected['color'].iloc[start:end]=='r')
#plt.plot(selected[0].iloc[start:end], markevery=list(markers_on),marker='X',mec='r',mew=np.log(np.log(dur))
#    ,ms=2*np.log(np.log(dur)),mfcalt='r')
#for i in range(5):
#    plt.plot(selected[i].iloc[start:end])
#    plt.show()
for j in [1,2,6,9]:
    print(j)
    plt.plot(list(selected[j].iloc[start:end].values))
#    plt.xlabel('timeslots',fontsize=28)
#    plt.ylabel('phase 1 current magnitude pmu="1024"',fontsize=28)
    for i in anoms:
        if (i*int(SampleNum/2)+1) in list(np.arange(start,end)):
            plt.axvspan(i*int(SampleNum/2), ((i+1)*int(SampleNum/2)+40), color='red', alpha=0.5)
    plt.show()

    
print('This is real ones')
    

    
for j in ['L3MAG','C3MAG','PC','QC']:
    print(j)
    plt.plot(list(selected_data[j].iloc[start:end].values))
#    plt.xlabel('timeslots',fontsize=28)
#    plt.ylabel('phase 1 current magnitude pmu="1024"',fontsize=28)
    for i in anoms:
        if (i*int(SampleNum/2)+1) in list(np.arange(start,end)):
            plt.axvspan(i*int(SampleNum/2), ((i+1)*int(SampleNum/2)+40), color='red', alpha=0.5)
    plt.show()
    
#plt.savefig('long.pdf', format='pdf', dpi=1200)
#plt.savefig('long %d.png' %dur)
#%%
dur_anoms=[]
for i in anoms:
    if (i*int(SampleNum/2)+1) in list(np.arange(start,end)):
        dur_anoms.append([i*int(SampleNum/2),((i+1)*int(SampleNum/2)+20)])
        plt.plot(selected[2].iloc[i*int(SampleNum/2)-20:((i+1)*int(SampleNum/2)+40)].values)
        plt.xlabel('timeslots',fontsize=28)
        plt.ylabel('phase 1 current magnitude pmu="1024"',fontsize=28)
#        plt.savefig('figures/event %d.png' %i)
#        plt.savefig('figures/event %d.pdf' %i, format='pdf', dpi=1200)
        plt.show()

print(dur_anoms)
print(len(dur_anoms))

#%%
# =============================================================================
# =============================================================================
# # subplot
# PMU
# =============================================================================
plt.subplot(2, 2, 1)
plt.plot(list(selected_data['L1MAG'].values))
plt.title('Real PMU data')
plt.ylabel('Real Voltage')
#plt.ylim([7100,7200])

plt.subplot(2, 2, 2)
plt.plot(list(selected_data['C1MAG'].values))
#plt.xlabel('time')
plt.ylabel('Real Current')
#plt.ylim([1,2])

plt.subplot(2, 2, 3)
plt.plot(list(selected_data['PA'].values))
#plt.title('Real PMU data')
plt.ylabel('Real ACtive Power')
plt.xlabel('time')
#plt.ylim([7100,7200])

plt.subplot(2, 2, 4)
plt.plot(list(selected_data['QA'].values))
#plt.title('Real PMU data')
plt.ylabel('Real Reactive Power')
plt.xlabel('time')
#plt.ylim([7100,7200])

plt.savefig('real.png')
plt.show()
#%%%
ss=preprocessing.scale(selected_data,axis=0)
plt.subplot(2, 2, 1)
plt.plot(ss[:,0])
plt.title('scaled PMU data')
plt.ylabel('scaled Voltage')
#plt.ylim([7100,7200])

plt.subplot(2, 2, 2)
plt.plot(ss[:,6])
#plt.xlabel('time')
plt.ylabel('scaled Current')
#plt.ylim([1,2])

plt.subplot(2, 2, 3)
plt.plot(ss[:,13])
#plt.title('scaled PMU data')
plt.ylabel('scaled ACtive Power')
plt.xlabel('time')
#plt.ylim([7100,7200])

plt.subplot(2, 2, 4)
plt.plot(ss[:,16])
#plt.title('scaled PMU data')
plt.ylabel('scaled Reactive Power')
plt.xlabel('time')
#plt.ylim([7100,7200])
plt.savefig('scale.png')
plt.show()
#plt.savefig('scale.png')























