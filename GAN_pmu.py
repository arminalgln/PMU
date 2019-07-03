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
#%%

def load_data(start,SampleNum,N):
         #read a pickle file
    pkl_file = open('CompleteOneDay.pkl', 'rb')
    selected_data = pickle.load(pkl_file)
    pkl_file.close()


    for pmu in selected_data:
        selected_data[pmu]=pd.DataFrame.from_dict(selected_data[pmu])

    train=selected_data['1224']['C1MAG'].iloc[0:N*SampleNum].values
    
    end=start+SampleNum

    pmu='1224'
    shift=int(SampleNum)
    
    train_data=[]
    for i in range(N):
        train_data.append(selected_data[pmu]['C1MAG'][start+i*shift:end+i*shift]-np.mean(selected_data[pmu]['C1MAG'][start+i*shift:end+i*shift]))

    x_train=np.array(train_data)
    
    # convert shape of x_train from (60000, 28, 28) to (60000, 784) 
    # 784 columns per row
    
    return x_train,train
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
    discriminator.add(Dense(units=1024,input_dim=40, activation='relu'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    
    discriminator.add(Dense(units=512, activation='relu'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    discriminator.add(Dense(units=256, activation='relu'))
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
    scale=5
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
X_train, selected = load_data(0,40,4000)
batch_count = X_train.shape[0] / batch_size
#%%
generator= create_generator()
discriminator= create_discriminator()
gan = create_gan(discriminator, generator)
#%%
def training(generator,discriminator,gan,epochs, batch_size=100):
    scale=5
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
            
        if e == 1 or e % 5 == 0:
           
            plot_generated_images(e, generator)
training(generator,discriminator,gan,200,128)
#%%
a=[]
for i in range(4000):
    a.append(discriminator.predict(X_train[i].reshape(1,40)))
a=np.array(a)

plt.plot(a.ravel())
plt.show()
#%%
threshold=0.3
print(np.where(a<threshold)[0].shape)
for i in np.where(a<threshold)[0]:
    plt.plot(X_train[i])
plt.show()
#%%
for i in np.where(sigmoid(a,True)<0.05)[0]:
#    print(i)
    plt.plot(X_train[i])
plt.show()
#%%
   
for i in np.intersect1d(np.where(a<threshold)[0],np.where(sigmoid(a,True)<0.05)[0]):
    print(i)
    plt.plot(X_train[i])
#%%
for i in range(4):
    plt.plot(X_train[i])
#%%
def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))
#%%
#%%  
event_pointer=np.where(a<threshold)[0]
k=X_train.ravel()
k.reshape(1,4000)
k=pd.DataFrame(k)
plt.plot(k[0:1000],color='r')
plt.plot(k[1000:2000],color='b')
plt.show()
        