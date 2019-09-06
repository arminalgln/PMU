# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras
from keras.layers import Dense, Dropout, Input, Activation,Embedding, LSTM, Reshape, CuDNNLSTM, UpSampling2D,Conv2D,Flatten,MaxPooling2D
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

#%%
event_file="data/Armin_Data/event_hand_standardized.pkl"
pkl_file = open(event_file, 'rb')
events = pkl.load(pkl_file)
pkl_file.close()
#%%
xtr=[]
ytr=[]
#day='July_03'
for day in events:
    for anom in events[day]:
    #    for i in range(120):
        xtr.append(events[day][anom])
#        ytr.appe
    
xtr=np.array(xtr)
xtr=xtr.reshape(-1,1,12,240)
#s=xtr.shape
#xtr=xtr.reshape(s[0],s[1],1)
#%%
def adam_optimizer():
    return adam(lr=0.0002, beta_1=0.5)
#%%

autoencoder = Sequential()

# Encoder Layers
autoencoder.add(Dense(1028,activation='relu', input_dim=12*240))
autoencoder.add(LeakyReLU(0.2))

autoencoder.add(Dense(512,activation='relu'))
autoencoder.add(LeakyReLU(0.2))

autoencoder.add(Dense(256,activation='relu'))
autoencoder.add(LeakyReLU(0.2))

autoencoder.add(Dense(32,activation='relu', name="latent_space"))
autoencoder.add(LeakyReLU(0.2))


# Decoder Layers
autoencoder.add(Dense(256,activation='relu'))
autoencoder.add(LeakyReLU(0.2))

autoencoder.add(Dense(512,activation='relu'))
autoencoder.add(LeakyReLU(0.2))

autoencoder.add(Dense(1028,activation='relu'))
autoencoder.add(LeakyReLU(0.2))

autoencoder.add(Dense(12*240,activation='relu'))
autoencoder.add(LeakyReLU(0.2))

autoencoder.summary()


#%%


"""
Combined Autoencoder with convolutional layers, fully connected layers and upsampling decoder
:return: model
"""
# Input
input_img = Input(shape=(1, 12, 240))
# Encoder
x = Conv2D(16,(3,3),
           activation='relu',
           padding='same',
           data_format='channels_first')(input_img)
x = Conv2D(16,(3,3),
           activation='relu',
           padding='same',
           data_format='channels_first')(x)
x = MaxPooling2D((2,2),
                 padding='same',
                 data_format='channels_first')(x) # Size 8x14x14
x = Conv2D(32,(3,3),
           activation='relu',
           padding='same',
           data_format='channels_first')(x)
x = Conv2D(32,(3,3),
           activation='relu',
           padding='same',
           data_format='channels_first')(x)
x = MaxPooling2D((2,2),
                 padding='same',
                 data_format='channels_first')(x) # Size 16x7x7
x = Flatten()(x)
x = Dense(256)(x)

code= Dense(32,name='latent_space')(x)
# Decoder
x = Dense(256)(code)
x = Dense(2880)(x)
x = Reshape((16,3,60))(x)
x = UpSampling2D((2, 2),
                 data_format='channels_first')(x)
x = Conv2D(32, (3, 3),
           activation='relu',
           padding='same',
           data_format='channels_first')(x)
x = Conv2D(32, (3, 3),
           activation='relu',
           padding='same',
           data_format='channels_first')(x)
x = UpSampling2D((2, 2),
                 data_format='channels_first')(x)  # Size 16x16x16
x = Conv2D(16, (3, 3),
           activation='relu',
           padding='same',
           data_format='channels_first')(x)
decoded = Conv2D(1, (3, 3),
           activation='relu',
           padding='same',
           data_format='channels_first')(x)

autoencoder = Model(input_img, decoded)

#%%
autoencoder.summary()
#%%
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('latent_space').output)
encoder.summary()


#%%
autoencoder.compile(optimizer='adam', loss='msle')
autoencoder.fit(xtr, xtr,
                epochs=100,
                batch_size=10,
                )
#%%
##
encoder.save('encoder_CNN_161632232_256_32dense_100_10.h5')
autoencoder.save('autoencoder_CNN_161632232_256_32dense_100_10.h5')
#%%
#encoder=load_model('encoder_dense102851225632.h5')
#autoencoder=load_model('autoencoder_dense102851225632.h5')
#%%
num_images = 10
#np.random.seed(42)
random_test_images = np.random.randint(xtr.shape[0], size=num_images)

encoded_imgs = encoder.predict(xtr)
decoded_imgs = autoencoder.predict(xtr)

#plt.figure(figsize=(18, 4))

for i, image_idx in enumerate(random_test_images):
    # plot original image
    ax = plt.subplot(3, num_images, i + 1)
    plt.plot(xtr[image_idx].reshape(12, 240)[3])
#    plt.imshow(xtr[image_idx].reshape(12, 240))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # plot encoded image
    ax = plt.subplot(3, num_images, num_images + i + 1)
    plt.imshow(encoded_imgs[image_idx].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot reconstructed image
    ax = plt.subplot(3, num_images, 2*num_images + i + 1)
    plt.plot(decoded_imgs[image_idx].reshape(12, 240)[3])
#    plt.imshow(decoded_imgs[image_idx].reshape(12, 240))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#%%
# =============================================================================
# =============================================================================
# # Classification
# =============================================================================
# =============================================================================
# =============================================================================
# calculate the latent space for each event
# =============================================================================


encoded_imgs = encoder.predict(xtr)
#%%
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)
#kmeans.labels_


for n_clusters in np.arange(16,25):
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(encoded_imgs)
    silhouette_avg = silhouette_score(encoded_imgs, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    #%%
kmeans = KMeans(n_clusters=10, random_state=0).fit(encoded_imgs)
kmeans.labels_
#%%
for num,k in enumerate(kmeans.labels_):
#    print(k)
    if k == 4:
        print(k)
        plt.plot(xtr[num].reshape(12,240)[3])
        plt.show()
#%%















