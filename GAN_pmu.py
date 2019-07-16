# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import keras
from keras.layers import Dense, Dropout, Input
from keras.models import Model,Sequential
from keras.datasets import mnist
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.optimizers import adam
import numpy as np
import tensorflow as tf
import os
import pickle
import operator
import math
import time

#%% 
def load_data(start,SampleNum,N):
         #read a pickle file
    pkl_file = open('CompleteOneDay.pkl', 'rb')
    selected_data = pickle.load(pkl_file)
    pkl_file.close()
    for pmu in selected_data:
        selected_data[pmu]=pd.DataFrame.from_dict(selected_data[pmu])

    select=selected_data['1224']['C1MAG'].iloc[0:int(N*SampleNum/2)].values
    
    end=start+SampleNum

    pmu='1224'
    shift=int(SampleNum/2)
    
    train_data=[]
    for i in range(N):
        train_data.append(selected_data[pmu]['C1MAG'][start+i*shift:end+i*shift]-np.mean(selected_data[pmu]['C1MAG'][start+i*shift:end+i*shift]))

    x_train=np.array(train_data)
    
    # convert shape of x_train from (60000, 28, 28) to (60000, 784) 
    # 784 columns per row
    
    return x_train,select
#X_train=load_data()
#print(X_train.shape)
#%%
def adam_optimizer():
    return adam(lr=0.0002, beta_1=0.5)
#%%
def create_generator():
    generator=Sequential()
    generator.add(Dense(units=256,input_dim=100))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=512))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=1024))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=40))
    
    generator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return generator
g=create_generator()
g.summary()

#%%
def create_discriminator():
    discriminator=Sequential()
    discriminator.add(Dense(units=1024,input_dim=40))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    
    discriminator.add(Dense(units=512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    discriminator.add(Dense(units=256))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Dense(units=1, activation='sigmoid'))
    
    discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return discriminator
d =create_discriminator()
d.summary()
#%%
def create_gan(discriminator, generator):
    discriminator.trainable=False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
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
batch_size=500
start,SampleNum,N=(0,40,1000)
X_train, selected = load_data(start,SampleNum,N)
batch_count = X_train.shape[0] / batch_size
#%%
generator= create_generator()
discriminator= create_discriminator()
gan = create_gan(discriminator, generator)
#%%
def training(generator,discriminator,gan,epochs, batch_size):
    scale=1
    for e in range(1,epochs+1 ):
        print("Epoch %d" %e)
        for _ in tqdm(range(batch_size)):
        #generate  random noise as an input  to  initialize the  generator
            noise= scale*np.random.normal(0,1, [batch_size, 100])
            
            # Generate fake MNIST images from noised input
            generated_images = generator.predict(noise)
            
            # Get a random set of  real images
            image_batch =X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)]
            
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
            y_gen = np.ones(batch_size)
            
            # During the training of gan, 
            # the weights of discriminator should be fixed. 
            #We can enforce that by setting the trainable flag
            discriminator.trainable=False
            
            #training  the GAN by alternating the training of the Discriminator 
            #and training the chained GAN model with Discriminator’s weights freezed.
            gan.train_on_batch(noise, y_gen)
            
#        if e == 1 or e % 5 == 0:
#           
#            plot_generated_images(e, generator)
batch_size=1000
epochnum=10
tic=time.clock()
training(generator,discriminator,gan,epochnum,batch_size)
toc=time.clock()
print(toc-tic)
#%%
reducedmean=[]
count=0
for i in X_train:
    if count%2==0:
        reducedmean.append(i)
    count+=1

reducedmean=np.array(reducedmean)
reducedmean=reducedmean.ravel()
plt.plot(reducedmean)
plt.savefig('reduced.png')
reducedmean=pd.DataFrame(reducedmean)

#%%
a=[]
count=0
for i in range(N):

    a.append(discriminator.predict(X_train[i].reshape(1,SampleNum)))
#%%
a=np.array(a)
#%%
fig_size = plt.rcParams["figure.figsize"]
 
 
# Set figure width to 12 and height to 9
fig_size[0] = 8
fig_size[1] = 6
plt.plot(a.ravel())
plt.show()


#%%
high=0.9999
low=0.14
fig_size = plt.rcParams["figure.figsize"]
 
 
# Set figure width to 12 and height to 9
fig_size[0] = 8
fig_size[1] = 6
anoms=np.union1d(np.where(a>high)[0], np.where(a<low)[0])
print(np.union1d(np.where(a>high)[0], np.where(a<low)[0]).shape)
for i in anoms :
#    print(i)
    plt.plot(X_train[i])
plt.show()

#%%
selected=pd.DataFrame(selected)

#%%
fig_size = plt.rcParams["figure.figsize"]
 
 
# Set figure width to 12 and height to 9
fig_size[0] = 8
fig_size[1] = 6
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
plt.plot(selected[0].iloc[start:end])
plt.xlabel('timeslots',fontsize=28)
plt.ylabel('phase 1 current magnitude pmu="1024"',fontsize=28)
for i in anoms:
    if (i*int(SampleNum/2)+1) in list(np.arange(start,end)):
        plt.axvspan(i*int(SampleNum/2), ((i+1)*int(SampleNum/2)+40), color='red', alpha=0.5)
#plt.savefig('long.pdf', format='pdf', dpi=1200)
#plt.savefig('long %d.png' %dur)
#%%
dur_anoms=[]
for i in anoms:
    if (i*int(SampleNum/2)+1) in list(np.arange(start,end)):
        dur_anoms.append([i*int(SampleNum/2),((i+1)*int(SampleNum/2)+20)])
        plt.plot(selected[0].iloc[i*int(SampleNum/2)-20:((i+1)*int(SampleNum/2)+40)].values)
        plt.xlabel('timeslots',fontsize=28)
        plt.ylabel('phase 1 current magnitude pmu="1024"',fontsize=28)
        plt.savefig('figures/event %d.png' %i)
        plt.savefig('figures/event %d.pdf' %i, format='pdf', dpi=1200)
        plt.show()

print(dur_anoms)
print(len(dur_anoms))

