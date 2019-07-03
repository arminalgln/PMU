# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 18:33:21 2019

@author: hamed
"""

# =============================================================================
# importing liberaries
# =============================================================================

import numpy as np
import tensorflow as tf
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import operator
import math
#%%
 #read a pickle file
pkl_file = open('CompleteOneDay.pkl', 'rb')
selected_data = pickle.load(pkl_file)
pkl_file.close()

#%%

for pmu in selected_data:
    selected_data[pmu]=pd.DataFrame.from_dict(selected_data[pmu])
    
#%%
# =============================================================================
#     normalize data
# =============================================================================

train=selected_data['1224'].iloc[0:100000].values
    
#%%
#power factor calculation
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)


# fit the model
clf = IsolationForest(behaviour='new', max_samples=100,
                      random_state=rng, contamination=0.01)
clf.fit(train)
y_pred_train = clf.predict(train)
#y_pred_test = clf.predict(X_test)
#y_pred_outliers = clf.predict(X_outliers)

#%%
# =============================================================================
# Different plots
SampleNum=40
start=10000
end=start+SampleNum
N=200
pmu='1224'
shift=int(SampleNum/2)


for i in range(N):
    plt.plot(np.arange(SampleNum),selected_data[pmu]['C1MAG'][start+i*shift:end+i*shift])
plt.show()
for i in range(N):
    plt.plot(np.arange(SampleNum),selected_data[pmu]['L1MAG'][start+i*shift:end+i*shift])
plt.show()
for i in range(N):
    plt.plot(np.arange(SampleNum),selected_data[pmu]['PA'][start+i*shift:end+i*shift])
plt.show()
for i in range(N):
    plt.plot(np.arange(SampleNum),selected_data[pmu]['QA'][start+i*shift:end+i*shift])
plt.show()
for i in range(N):
    plt.plot(np.arange(SampleNum),selected_data[pmu]['pfA'][start+i*shift:end+i*shift])
plt.show()

plt.plot(selected_data[pmu]['C1MAG'][start:end+N*shift])
plt.show()
plt.scatter(selected_data[pmu]['C1MAG'][start:end+N*shift],selected_data[pmu]['L1MAG'][start:end+N*shift])

#%%

SampleNum=40
start=10000
end=start+SampleNum
N=2000
pmu='1224'
shift=int(SampleNum/2)

from sklearn import preprocessing

for i in range(N):
    plt.plot(np.arange(SampleNum),selected_data[pmu]['C1MAG'][start+i*shift:end+i*shift]-np.mean(selected_data[pmu]['C1MAG'][start+i*shift:end+i*shift]))
plt.show()
for i in range(N):
    plt.plot(np.arange(SampleNum),selected_data[pmu]['L1MAG'][start+i*shift:end+i*shift]-np.mean(selected_data[pmu]['L1MAG'][start+i*shift:end+i*shift]))
plt.show()
for i in range(N):
    plt.plot(np.arange(SampleNum),selected_data[pmu]['PA'][start+i*shift:end+i*shift]-np.mean(selected_data[pmu]['PA'][start+i*shift:end+i*shift]))
plt.show()
for i in range(N):
    plt.plot(np.arange(SampleNum),selected_data[pmu]['QA'][start+i*shift:end+i*shift]-np.mean(selected_data[pmu]['QA'][start+i*shift:end+i*shift]))
plt.show()
for i in range(N):
    plt.plot(np.arange(SampleNum),selected_data[pmu]['pfA'][start+i*shift:end+i*shift]-np.mean(selected_data[pmu]['pfA'][start+i*shift:end+i*shift]))
plt.show()

plt.plot(selected_data[pmu]['C1MAG'][start:end+N*shift])
plt.show()
plt.scatter(selected_data[pmu]['C1MAG'][start:end+N*shift],selected_data[pmu]['L1MAG'][start:end+N*shift])


#%%


fig, ax1 = plt.subplots()


SampleNum=200
start=0
end=start+SampleNum

color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('exp', color=color)
ax1.plot(selected_data['1224']['C1MAG'][start:end], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
ax2.plot(selected_data['1224']['L1MAG'][start:end], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
#%%

# =============================================================================
# dividing dataset to the sub sequence of time series.
# =============================================================================

SampleNum=40
start=10000
end=start+SampleNum
N=20000
pmu='1224'
shift=int(SampleNum/2)

train_data=[]
for i in range(N):
    train_data.append(selected_data[pmu]['C1MAG'][start+i*shift:end+i*shift]-np.mean(selected_data[pmu]['C1MAG'][start+i*shift:end+i*shift]))


#%%
n_clusters=100
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters, random_state=0).fit(train_data)

#%%
cm=kmeans.cluster_centers_


for i in range(N):
    plt.plot(np.arange(SampleNum),selected_data[pmu]['C1MAG'][start+i*shift:end+i*shift]-np.mean(selected_data[pmu]['C1MAG'][start+i*shift:end+i*shift]))
plt.show()
#%%
for i in range(n_clusters):
    plt.plot(cm[i])
plt.show()
#%%
import collections
labels=kmeans.labels_
groups=collections.Counter(labels)

anomalies={}
counter=0
anom=[]
for i in range(n_clusters):
    if groups[i] <= 3:
        index=np.where(labels==i)
        anomalies[counter]=index
        for i in index[0]:
            anom.append(i)
        counter+=1
#for i in range(N):
        
anom=np.array(anom)
anom=np.sort(anom)
anom_unique=[]
for i in range(int(anom.shape[0]-1)):
    if anom[i+1]-anom[i]!=1:
        anom_unique.append(anom[i+1])

#%%
for i in anom_unique:
    plt.plot(np.arange(SampleNum),train_data[i])
plt.show()

for i in anom_unique:
    plt.plot(train_data[i])
plt.show()
#%%
for i in range(N):
    plt.plot(selected_data[pmu]['C1MAG'][start+i*shift:end+i*shift]-np.mean(selected_data[pmu]['C1MAG'][start+i*shift:end+i*shift]))
plt.show()
#%%
for i in anom_unique:
    plt.plot(np.arange(SampleNum),train_data[i])
    plt.show()

#%%
ypred=kmeans.predict(train_data)
pred=[]
for i in ypred:
    pred.append(list(cm[i]))
    
pred=np.array(pred)
pred=pred.ravel()

tr=np.array(train_data)
tr=tr.ravel()

diff=tr-pred

plt.plot(tr)

#%%
from sklearn.cluster import KMeans
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
x = x.reshape((x.shape[0], -1))
x = np.divide(x, 255.)
# 10 clusters
n_clusters = len(np.unique(y))
# Runs in parallel 4 CPUs
kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
# Train K-Means.
y_pred_kmeans = kmeans.fit_predict(x)
# Evaluate the K-Means clustering accuracy.
metrics.acc(y, y_pred_kmeans)
#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

np.random.seed(1337)

# MNIST dataset
data=np.array(train_data)
trnum=int(N*0.95)
testnum=data.shape[0]-trnum
#x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
#x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = data[0:trnum]
x_test = data[trnum:data.shape[0]]
x_train=x_train.reshape((trnum,data.shape[1],1,1))
x_test=x_test.reshape((testnum,data.shape[1],1,1))

image_size = data.shape[1]
# Generate corrupted MNIST images by adding noise with normal dist
# centered at 0.5 and std=0.5

# Network parameters
input_shape = (image_size, 1,1)
batch_size = 128
kernel_size = 3
latent_dim = 16
# Encoder/Decoder number of CNN layers and filters per layer
layer_filters = [32, 64]

# Build the Autoencoder Model
# First build the Encoder Model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
# Stack of Conv2D blocks
# Notes:
# 1) Use Batch Normalization before ReLU on deep networks
# 2) Use MaxPooling2D as alternative to strides>1
# - faster but not as good as strides>1
for filters in layer_filters:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               activation='relu',
               padding='same')(x)

# Shape info needed to build Decoder Model
shape = K.int_shape(x)

# Generate the latent vector
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)

# Instantiate Encoder Model
encoder = Model(inputs, latent, name='encoder')
encoder.summary()

# Build the Decoder Model
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(40)(latent_inputs)
x = Reshape((40,1,1))(x)

# Stack of Transposed Conv2D blocks
# Notes:
# 1) Use Batch Normalization before ReLU on deep networks
# 2) Use UpSampling2D as alternative to strides>1
# - faster but not as good as strides>1
for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=1,
                        activation='relu',
                        padding='same')(x)

x = Conv2DTranspose(filters=1,
                    kernel_size=kernel_size,
                    padding='same')(x)

outputs = Activation('tanh', name='decoder_output')(x)
#outputs=x
# Instantiate Decoder Model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# Autoencoder = Encoder + Decoder
# Instantiate Autoencoder Model
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()

autoencoder.compile(loss='mse', optimizer='adam')

# Train the autoencoder
autoencoder.fit(x_train,
                x_train,
                validation_data=(x_test, x_test),
                epochs=30,
                batch_size=batch_size)
#%%
# Predict the Autoencoder output from corrupted test images
x_decoded = autoencoder.predict(x_test)
#%%/
# Display the 1st 8 corrupted and denoised images
rows, cols = 2, 6
num = rows * cols
imgs = np.concatenate([x_test[:num], x_test_noisy[:num], x_decoded[:num]])
imgs = imgs.reshape((rows * 3, cols, image_size, image_size))
imgs = np.vstack(np.split(imgs, rows, axis=1))
imgs = imgs.reshape((rows * 3, -1, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.title('Original images: top rows, '
          'Corrupted Input: middle rows, '
          'Denoised Input:  third rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
Image.fromarray(imgs).save('corrupted_and_denoised.png')
plt.show()

#%%
main=[]
decode=[]
for i in range(testnum):
    main.append(x_test[i].ravel())
    decode.append(x_decoded[i].ravel())

print(len(decode[0]))
main=np.array(main)
decode=np.array(decode)
print((decode.shape))
main=main.reshape(testnum*40)
decode=decode.reshape(testnum*40)
print((decode.shape))
plt.plot(main)

plt.plot(decode)
#%%
start=15600
dur=200
end=start+dur
plt.plot(main[start:end])
plt.plot(decode[start:end])
plt.show()

plt.plot(main-decode)

#%%
x_decoded = autoencoder.predict(x_train)
main=[]
decode=[]
for i in range(trnum):
    main.append(x_train[i].ravel())
    decode.append(x_decoded[i].ravel())

print(len(decode[0]))
main=np.array(main)
decode=np.array(decode)
print((decode.shape))
main=main.reshape(trnum*40)
decode=decode.reshape(trnum*40)
print((decode.shape))
plt.plot(main)

plt.plot(decode)

plt.show()

plt.plot(main-decode)
#%%
start=15600
dur=200
end=start+dur
plt.plot(main[start:end])
plt.plot(decode[start:end])
plt.show()

plt.plot(main-decode)
