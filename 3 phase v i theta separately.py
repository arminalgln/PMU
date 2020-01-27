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
        dir="data/Armin_Data/July_0"+str(n)+"/"
    else:
        dir="data/Armin_Data/July_"+str(n)+"/"
#dir='data/Armin_Data/July_03'

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
        
#        cosin['TA']=np.cos((selected_data['L1Ang']-selected_data['C1Ang'])*(np.pi/180))
#        cosin['TB']=np.cos((selected_data['L2Ang']-selected_data['C2Ang'])*(np.pi/180))
#        cosin['TC']=np.cos((selected_data['L3Ang']-selected_data['C3Ang'])*(np.pi/180))
            
    #    Reacive['A']=selected_data['L1Mag']*selected_data['C1Mag']*(np.sin((selected_data['L1Ang']-selected_data['C1Ang'])*(np.pi/180)))
    #    Reacive['B']=selected_data['L2Mag']*selected_data['C2Mag']*(np.sin((selected_data['L2Ang']-selected_data['C2Ang'])*(np.pi/180)))
    #    Reacive['C']=selected_data['L3Mag']*selected_data['C3Mag']*(np.sin((selected_data['L3Ang']-selected_data['C3Ang'])*(np.pi/180)))
        #   
        #pf['A']=Active['A']/np.sqrt(np.square(Active['A'])+np.square(Reacive['A']))
        #pf['B']=Active['B']/np.sqrt(np.square(Active['B'])+np.square(Reacive['B']))
        #pf['C']=Active['C']/np.sqrt(np.square(Active['C'])+np.square(Reacive['C']))
        
        
#        selected_data['TA']=cosin['TA']
#        selected_data['TB']=cosin['TB']
#        selected_data['TC']=cosin['TC']
        
        selected_data=selected_data.drop(columns=['Unnamed: 0'])
    
    #    
    #    selected_data['QA']=Reacive['A']
    #    selected_data['QB']=Reacive['B']
    #    selected_data['QC']=Reacive['C'] 
    #    
        if count==0:
            whole_data=selected_data.values
        else:
            whole_data=np.append(whole_data,selected_data.values,axis=0)
            
    k=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG','L1Ang','L2Ang','L3Ang','C1Ang','C2Ang','C3Ang']
        
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
    pickle.dump(day_data, output)
    output.close()
    print(n)

#%%

# =============================================================================
# =============================================================================
# # train data prepreation
# =============================================================================
# =============================================================================
filename='data/Armin_Data/July_03/pkl/julseppf3.pkl'
k=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG','TA', 'TB', 'TC']
#%%
dds=load_standardized_data_with_features(filename,k)
#%%
dd=load_data_with_features(filename,k)

#%%
start,SampleNum,N=(0,40,500000)
filename='data/Armin_Data/July_03/pkl/julseppf3.pkl'
k=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG','TA', 'TB', 'TC']
tt=load_train_vitheta_data_V(start,SampleNum,N,filename,k)
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
batch_size=10
epochnum=20


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
kk=['L1mag']
for idx,key in enumerate(kk):
    X_train_temp=X_train[:,(idx+6)]
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
    #%%
scores_temp={}
probability_mean={}
anomalies_temp={}
#kk=['TA','TB','TC']
for idx,key in enumerate(kk):
    print(key)
    X_train_temp=X_train[:,(idx+6)]
#X_train.reshape(N,3*SampleNum)
    X_train_temp=X_train_temp.reshape(N,SampleNum,1)

#    id=int(np.floor(idx/3))
#    mode=k[id*3]
#    dis_name='dis_sep_onelearn_'+mode+'.h5'
#    
#    discriminator=load_model(dis_name)
    
    
    rate=1000
    shift=N/rate
    scores_temp[key]=[]
    for i in range(rate-1):
        temp=discriminator.predict_on_batch(X_train_temp[int(i*shift):int((i+1)*shift)])
        scores_temp[key].append(temp)
        print(i)
    
    scores_temp[key]=np.array(scores_temp[key])
    scores_temp[key]=scores_temp[key].ravel()
    
    probability_mean[key]=np.mean(scores_temp[key])
    data=scores_temp[key]-probability_mean[key]
    
    mu, std = norm.fit(data)
    
    zp=3
    
    high=mu+zp*std
    low=mu-zp*std
    
    anomalies_temp[key]=np.union1d(np.where(data>=high)[0], np.where(data<=low)[0])
    print(anomalies_temp[key].shape)
#%%
kk=['L1MAG','C1MAG','TA']
for idx,key in enumerate(kk):
    X_train_temp=X_train[:,idx*3]
#X_train.reshape(N,3*SampleNum)
    X_train_temp=X_train_temp.reshape(N,SampleNum,1)
    tic = time.clock()   
    training(generator,discriminator,gan,epochnum,batch_size)
    toc = time.clock()
    print(toc-tic)
    
    gan_name='gan_sep_onelearn_'+key+'.h5'
    gen_name='gen_sep_onelearn_'+key+'.h5'
    dis_name='dis_sep_onelearn_'+key+'.h5'
    print(dis_name)
    gan.save(gan_name)
    generator.save(gen_name)
    discriminator.save(dis_name)
#%%
    
scores={}
probability_mean={}
anomalies={}
#k=k[0:3]
for idx,key in enumerate(k):
    print(key)
    X_train_temp=X_train[:,idx]
#X_train.reshape(N,3*SampleNum)
    X_train_temp=X_train_temp.reshape(N,SampleNum,1)

    id=int(np.floor(idx/3))
    mode=k[id*3]
    dis_name='dis_sep_onelearn_'+mode+'.h5'
    print(dis_name)
    
    discriminator=load_model(dis_name)
    
    
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
def check_common(F1,F2):
    common=[]
    for event in F1:
        shift_events=[event-2,event-1,event,event+1,event+2]
        
        for i in shift_events:
            if i in F2 and i not in common:
                common.append(i)
    common=np.array(common)
    return common

#%%
commons={}
uni=np.array([])
for idx1,F1 in enumerate(anomalies):
    
    for idx2,F2 in enumerate(anomalies):
        commons[F1+'_'+F2]=check_common(anomalies[F1],anomalies[F2])
        uni=np.union1d(uni,np.union1d(anomalies[F1],anomalies[F2]))
        
            
#%%
def show(events):
    SampleNum=40
    for anom in events:
            anom=int(anom)
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
            plt.title('T')    
            
            plt.subplot(224)
            for i in [9,10,11]:
                plt.plot(select_1224[i][anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)])
            plt.legend('A' 'B' 'C')
            plt.title('Q')    
            plt.show()
#%%
def check_event_in_feature(event,f):
    out=0
    shift_events=[event-2,event-1,event,event+1,event+2]
    
    for i in shift_events:
            if i in f:
                    out=1
    
    return out
#%%
# =============================================================================
# each detected event should have a vector of detected feature
# =============================================================================
    
def event_vector(event,anomalies):
    vector=np.zeros((9,1))
    for idx,f in enumerate(anomalies):
        vector[idx,0]=check_event_in_feature(event,anomalies[f])
        
    return vector
#%%
event_vectors={}
for id,event in enumerate(uni):
    print(id)
    event_vectors[event]=event_vector(event,anomalies)
#%%
# =============================================================================
# unique events
# =============================================================================

def unique_events(uni):
    unique=[42]
    for i in uni:
        out=1
        shift_events=[i-2,i-1,i,i+1,i+2]
        print(shift_events)
        for j in shift_events:
            if j in unique:
                out=0
        
        if out==1:
            unique.append(i)
            
    unique=np.array(unique)
    return unique
#%%
uniques=unique_events(uni)
#%%
# =============================================================================
# two group close intersection check
# =============================================================================
def two_check_inter(g1,g2):
    intersection=[]
    
    for i in g1:
        shift_events=[i-2,i-1,i,i+1,i+2]
        for j in g2:
            
            if j in shift_events and j not in intersection:
                intersection.append(i)
    intersection=np.array(intersection)
    
    return intersection

#%%
cluster_vecotrs=[]
cluster_vecotrs_events=[]
for i,k in enumerate(event_vectors):
    cluster_vecotrs.append(event_vectors[k])
    cluster_vecotrs_events.append(k)

cluster_vecotrs=np.array(cluster_vecotrs)
cluster_vecotrs_events=np.array(cluster_vecotrs_events)
#%%
# =============================================================================
# cluster events based on detected features
# =============================================================================

def feature_clustering(event_vectors):
    
    cluster_vecotrs=[]
    cluster_vecotrs_events=[]
    for i,k in enumerate(event_vectors):
        cluster_vecotrs.append(event_vectors[k])
        cluster_vecotrs_events.append(k)
    
    cluster_vecotrs=np.array(cluster_vecotrs)
    cluster_vecotrs_events=np.array(cluster_vecotrs_events)
    
    unique_feature_clusters=np.unique(cluster_vecotrs,axis=0)
    
    feature_clusters={}
    
    for i in range(unique_feature_clusters.shape[0]):
        print(i)
        feature_clusters[i]=[]
        for j in event_vectors:
            if list(unique_feature_clusters[i].ravel())==list(event_vectors[j].ravel()):
                feature_clusters[i].append(j)
    
    return feature_clusters
#%%
for i in range(197):
    print(list(unique_feature_clusters[i])) 
    show([ff[i][0]])
#%%
i=190
print(list(unique_feature_clusters[i]))
show([ff[i][0]])
#%%
for j in ff[i]:
    show([j])


#%%
pkl_file = open('data/Armin_data/oneday_3d_events.pkl', 'rb')
whole_features = pkl.load(pkl_file)
pkl_file.close()
# =============================================================================
# =============================================================================
# # selected data features for final detection
# =============================================================================
# =============================================================================
data=[]
#xt, lmbda = stats.boxcox((whole_features['scores'])+1)
#xt=preprocessing.scale(xt)
#data.append(whole_features['scores'])
#
##xt, lmbda = stats.boxcox((whole_features['scores_V'])+1)
##xt=preprocessing.scale(xt)
#data.append(whole_features['scores_V'])

xt, lmbda = stats.boxcox((whole_features['maxvar']))
#xt=preprocessing.scale(xt)
data.append(xt)

xt, lmbda = stats.boxcox((whole_features['maxmaxmin']))
#xt=preprocessing.scale(xt)
data.append(xt)

data=np.array(data)

# =============================================================================
# =============================================================================
# #     basic whole anomalies with zp=3
# =============================================================================
# =============================================================================
zp=3

names=['maxvar','maxmaxmin']
basic_anoms={}
for i,d in enumerate(data):
    dt = d
    # Fit a normal distribution to the data:
    mu, std = norm.fit(dt)
    
    high=mu+zp*std
    low=mu-zp*std
    anoms_1224=np.union1d(np.where(dt>=high)[0], np.where(dt<=low)[0])
    print(anoms_1224.shape)
    basic_anoms[names[i]]=anoms_1224

    #%%
# =============================================================================
# anomaly flag and color
# =============================================================================

flag=np.zeros((scores['L1MAG'].shape[0],1))
color=["b" for x in range(scores['L1MAG'].shape[0])]

flag_mvmpm=np.zeros((scores['L1MAG'].shape[0],1))
color_mvmpm=["b" for x in range(scores['L1MAG'].shape[0])]
for i in uni:
    flag[int(i)]=1
    color[int(i)]="r"
    flag_mvmpm[int(i)]=1
    color_mvmpm[int(i)]="r"

for i in basic_anoms:
    for j in basic_anoms[i]:
        if j<499500:
            flag_mvmpm[int(j)]=1
            color_mvmpm[int(j)]="r"
#%%
# =============================================================================
# =============================================================================
# # 3d catter plot
# =============================================================================
# =============================================================================
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(scores['L1MAG'], whole_features['maxmaxmin_scale'][0:scores['L1MAG'].shape[0]], whole_features['maxvar_scale'][0:scores['L1MAG'].shape[0]],color=color)


ax.set_xlabel('MPM')
ax.set_ylabel('MV')
ax.set_zlabel('Scaled GAN scores')
#%%
high_event_vectors_dict={}
high_event_vectors=[]

for i in event_vectors:
    vec=[]
    if sum(event_vectors[i][0:3])!=0:
        vec.append(1)
    else:
        vec.append(0)
        

    if sum(event_vectors[i][3:6])!=0:
        vec.append(1)
    else:
        vec.append(0)
        
    if sum(event_vectors[i][6:9])!=0:
        vec.append(1)
    else:
        vec.append(0)

    if sum(event_vectors[i][0:3])>=2 or sum(event_vectors[i][3:6])>=2 or sum(event_vectors[i][6:9])>=2:
        vec.append(1)
    else:
        vec.append(0)

    high_event_vectors_dict[i]=vec
    high_event_vectors.append(vec)
#%%
selected_events_for_clustering=[]
for e in high_event_vectors_dict:
    if sum(high_event_vectors_dict[e])>=3:
        selected_events_for_clustering.append(e)
selected_events_for_clustering=np.array(selected_events_for_clustering)
#%%







