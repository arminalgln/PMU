import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
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
from scipy.spatial.distance import euclidean
from tslearn.clustering import GlobalAlignmentKernelKMeans
import loading_data
from loading_data import load_real_data, load_standardized_data,load_train_data,load_train_data_V
from scipy import stats
from sklearn.ensemble import IsolationForest

#%%
# =============================================================================
# =============================================================================
# # select the desired day standardized data
# =============================================================================
# =============================================================================
filename='data/Armin_Data/July_07/pkl/J7.pkl'
#%%
selected=load_standardized_data(filename)
#%%
# =============================================================================
# =============================================================================
# # load the best GAN model
# =============================================================================
# =============================================================================
gan=load_model('GPU_gan_mul_LSTM_twolayer_N500000_e1000_b100.h5')
generator=load_model('GPU_generator_mul_LSTM_twolayer_N500000_e1000_b100.h5')
discriminator=load_model('GPU_discriminator_mul_LSTM_twolayer_N500000_e1000_b100.h5')

#%%
# =============================================================================
# =============================================================================
# # Load training data
# =============================================================================
# =============================================================================
start,SampleNum,N,filename=(0,40,500000,filename)
#%%
X_train= load_train_data(start,SampleNum,N,filename)
#%%
X_train=X_train.reshape(N,12*SampleNum)
X_train=X_train.reshape(N,SampleNum,12)

rate=1000
shift=N/rate
scores=[]
for i in range(rate):
    temp=discriminator.predict_on_batch(X_train[int(i*shift):int((i+1)*shift)])
    scores.append(temp)
    print(i)

scores=np.array(scores)
scores=scores.ravel()

probability_mean=np.mean(scores)
a=scores-probability_mean

#%%
ganV=load_model('GPU_gan_voltage_N500000_e100_b10_1225.h5')
generatorV=load_model('GPU_generator_voltage_N500000_e100_b10_1225.h5')
discriminatorV=load_model('GPU_discriminator_voltage_N500000_e1000_b10_1225.h5')
#%%
start,SampleNum,N,filename=(0,40,500000,filename)
#%%
X_train_V= load_train_data_V(start,SampleNum,N,filename)

#%%
X_train_V=X_train_V.reshape(N,3*SampleNum)
X_train_V=X_train_V.reshape(N,SampleNum,3)

rate=1000
shift=N/rate
scoresV=[]
for i in range(rate):
    temp=discriminatorV.predict_on_batch(X_train_V[int(i*shift):int((i+1)*shift)])
    scoresV.append(temp)
    print(i)

scoresV=np.array(scoresV)
scoresV=scoresV.ravel()

probability_meanV=np.mean(scoresV)
aV=scoresV-probability_meanV
#%%
whole_features={}
whole_features['scores']=[]
whole_features['maxmin']=[]
whole_features['var']=[]

for i in range(N):
    maxmin=[]
    var=[]
    for j in range(12):
        maxmin.append(np.max(X_train[i][:,j])-np.min(X_train[i][:,j]))
        var.append(np.var(X_train[i][:,j]))
    
    whole_features['scores'].append(a[i])
    whole_features['maxmin'].append(maxmin)
    whole_features['var'].append(var)
    if i% 10000==0:
        print('iter num: %i', i)
        
#%%
    
whole_features['scores']=np.array(whole_features['scores'])
whole_features['maxmin']=np.array(whole_features['maxmin'])
whole_features['var']=np.array(whole_features['var'])
#%%
whole_features['scores_V']=scoresV
#%%
whole_features['scores_scale_V']=preprocessing.scale(aV)

#%%
whole_features['maxmaxmin']=np.max(whole_features['maxmin'],axis=1)
whole_features['maxvar']=np.max(whole_features['var'],axis=1)
#%%
whole_features['scores_scale']=preprocessing.scale(whole_features['scores'])
#%%
# =============================================================================
# mark the anomalies
# =============================================================================
excel_file='figures/all_events/July_03/GAN/anoms_July_03.csv'
anomalies=pd.read_csv(excel_file,header=None)[0]

#%%
# =============================================================================
# =============================================================================
# # event_points come from "model event detection accurcy .py"
# =============================================================================
# =============================================================================
event_points[3]['GAN_total_events']
#%%
whole_features['anoms']=np.zeros((N,1))
for i in event_points[3]['GAN_total_events']:
    i=int(float(i))
    whole_features['anoms'][i]=1
#%%
an=0
whole_features['color']=[]
for i in whole_features['anoms']:
#    print(i)
    if int(i) == 0:
        whole_features['color'].append('b')
    else:
        an+=1
        whole_features['color'].append('r')
        
whole_features['color']=np.array(whole_features['color'])
print(an) 
#%%

output = open('data/Armin_data/oneday_3d_events.pkl', 'wb')
pkl.dump(whole_features, output)
output.close()

#%%
pkl_file = open('data/Armin_data/oneday_3d_events.pkl', 'rb')
whole_features = pkl.load(pkl_file)
pkl_file.close()
#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(whole_features['maxmaxmin'], whole_features['maxvar'], whole_features['scores_scale'],color=whole_features['color'])


ax.set_xlabel('MPM')
ax.set_ylabel('MV')
ax.set_zlabel('Scaled GAN scores')


    #%%
blue_index=[np.where((0.04 <= whole_features['maxvar'][0:10000]) & (whole_features['maxvar'][0:10000] <= 0.05))]
#%%
X=np.zeros((N,4))
X[:,0]=whole_features['scores_scale']
X[:,3]=whole_features['scores_scale_V']
X[:,1]=whole_features['maxmaxmin']
X[:,2]=whole_features['maxvar']
#%%
rng = np.random.RandomState(42)
clf = IsolationForest(behaviour='new', max_samples=1000,
                      random_state=rng, contamination='auto')
clf.fit(X)
y_pred_train = clf.predict(X)
    
    
#%%
for anom in blue_index[0][0]:
        print(anom)
        anom=int(anom)
        
        plt.subplot(221)
        for i in [0,1,2]:
            plt.plot(selected[i][anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)])
        plt.legend('A' 'B' 'C')
        plt.title('V')
            
        plt.subplot(222)
        for i in [3,4,5]:
            plt.plot(selected[i][anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)])
        plt.legend('A' 'B' 'C')
        plt.title('I')  
        
        plt.subplot(223)
        for i in [6,7,8]:
            plt.plot(selected[i][anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)])
        plt.legend('A' 'B' 'C') 
        plt.title('P')    
        
        plt.subplot(224)
        for i in [9,10,11]:
            plt.plot(selected[i][anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)])
        plt.legend('A' 'B' 'C')
        plt.title('Q')    
        plt.show()
#%%
whole_features['maxmaxmin_scale']=preprocessing.scale(whole_features['maxmaxmin'])
whole_features['maxvar_scale']=preprocessing.scale(whole_features['maxvar'])
#%%
lamb=3
data =np.log(whole_features['maxvar'])

# Fit a normal distribution to the data:
mu, std = norm.fit(data)

# Plot the histogram.
plt.hist(data, bins=1000, density=True, alpha=0.6, color='g')

# Plot the PDF.
#xmin, xmax = plt.xlim()
#
#x = np.linspace(xmin, xmax, 100)
#p = norm.pdf( mu, std)
#plt.plot(p, 'k', linewidth=2)
#title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
#plt.title(title)


plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
#%%
#X_train=np.zeros((N,2))
#X_train[:,0]=np.log(whole_features['maxmaxmin'])
#X_train[:,1]=np.log(whole_features['maxvar'])
data=np.log(whole_features['scores_V']).reshape(-1,1)

#for i in range(10):
n=2
clf = mixture.GaussianMixture(n_components=n, covariance_type='full')
clf.fit(data)
print(clf.bic(data))
for i in range(n):
    print(clf.means_[i][0]-3*clf.covariances_[i][0][0],clf.means_[i][0]+3*clf.covariances_[i][0][0])
print(clf.means_)
print(clf.covariances_)
#%%

#np.prod(clf.covariances_)
#np.mean(clf.means_)
#
from scipy import stats
# Plot the histogram.
data=(whole_features['maxvar']).reshape(-1,1)
#data=data-np.mean(data)
data=stats.boxcox(data)
data=np.log(data)
lamb=3
data=(data-1)**lamb/lamb
#mu, std = norm.fit(data)
plt.hist(data[0], bins=1000, density=True, alpha=0.6, color='g')
#%%
# =============================================================================
# =============================================================================
# # plot the histogram of different features
# =============================================================================
# =============================================================================
from scipy import stats
#%%

fig = plt.figure()
ax1 = fig.add_subplot(221)
data=(whole_features['maxvar'])
xt, lmbda = stats.boxcox(data)
prob = stats.probplot(data, dist=stats.norm, plot=ax1)
ax1.set_xlabel('MV before transformation')
ax1.set_title('')
#ax1.set_title('Probplot after Yeo-Johnson transformation')

ax2 = fig.add_subplot(222)
data=(whole_features['maxmaxmin'])
xt, lmbda = stats.boxcox(data)
prob = stats.probplot(data ,dist=stats.norm, plot=ax2)
ax2.set_title('')
ax2.set_xlabel('MPM before transformation')


ax3 = fig.add_subplot(223)
data=(whole_features['maxvar'])
xt, lmbda = stats.boxcox(data)
prob = stats.probplot(xt, dist=stats.norm, plot=ax3)
ax3.set_title('')
ax3.set_xlabel('MV after transformation')

#ax1.set_title('Probplot after Yeo-Johnson transformation')

ax4 = fig.add_subplot(224)
data=(whole_features['maxmaxmin'])
xt, lmbda = stats.boxcox(data)
prob = stats.probplot(xt ,dist=stats.norm, plot=ax4)
ax4.set_title('')
ax4.set_xlabel('MPM after transformation')

#fig.suptitle('Probability Plot')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
#plt.savefig('figures/paper/before_after_transformation.',dpi=300, bbox_inches='tight')
#%%
data=[]
#xt, lmbda = stats.boxcox((whole_features['scores'])+1)
#xt=preprocessing.scale(xt)
data.append(whole_features['scores'])

#xt, lmbda = stats.boxcox((whole_features['scores_V'])+1)
#xt=preprocessing.scale(xt)
data.append(whole_features['scores_V'])

xt, lmbda = stats.boxcox((whole_features['maxvar']))
#xt=preprocessing.scale(xt)
data.append(xt)

xt, lmbda = stats.boxcox((whole_features['maxmaxmin']))
#xt=preprocessing.scale(xt)
data.append(xt)

data=np.array(data)
#%%
mean = np.mean(data,axis=1)
cov = np.cov(data)
#%%
data=[]
xt, lmbda = stats.boxcox((whole_features['scores'])+1)
xt=preprocessing.scale(xt)
data.append(xt)

xt, lmbda = stats.boxcox((whole_features['scores_V'])+1)
xt=preprocessing.scale(xt)
data.append(xt)

xt, lmbda = stats.boxcox((whole_features['maxvar']))
xt=preprocessing.scale(xt)
data.append(xt)

xt, lmbda = stats.boxcox((whole_features['maxmaxmin']))
xt=preprocessing.scale(xt)
data.append(xt)

data=np.array(data)
#%%
mean = np.mean(data,axis=1)
cov = np.cov(data)
rv=multivariate_normal(mean,cov)
x=np.transpose(data)
y=rv.pdf(x)
#%%
#%%
# =============================================================================
# =============================================================================
# # extract the anomalies wrt each feature
# =============================================================================
# =============================================================================
zp=2

names=['scores','scores_V','maxvar','maxmaxmin']
anoms={}
for i in names:
    anoms[i]=[]
for zp in np.arange(2.5,5,0.1):
    for i,d in enumerate(data):
        dt = d
        # Fit a normal distribution to the data:
        mu, std = norm.fit(dt)
        
        high=mu+zp*std
        low=mu-zp*std
        anoms_1224=np.union1d(np.where(dt>=high)[0], np.where(dt<=low)[0])
        print(anoms_1224.shape)
        anoms[names[i]].append(anoms_1224.shape)
#%%
# =============================================================================
# =============================================================================
# # different Zp
# =============================================================================
# =============================================================================
zp=np.arange(2.5,5,0.1)
for i in anoms:
    plt.plot(zp,anoms[i])
plt.yticks(fontsize=15)
plt.legend(('GAN', 'GANV', 'MV', 'MP'),fontsize= 20)
#        plt.figtext(.5,.9,'Temperature', fontsize=100, ha='center')
plt.xlabel('Thresold (Zp)',fontsize= 30)
plt.ylabel('Number of detected aevents',fontsize= 30)
plt.show()
#%%
filename='data/Armin_Data/July_03/pkl/J3.pkl'
select_1224=load_real_data(filename)
    #%%
zp=3.1
anoms31={}
for i,d in enumerate(data):
    dt = d
    # Fit a normal distribution to the data:
    mu, std = norm.fit(dt)
    
    high=mu+zp*std
    low=mu-zp*std
    anoms_1224=np.union1d(np.where(dt>=high)[0], np.where(dt<=low)[0])
    print(anoms_1224.shape)
    anoms31[names[i]]=anoms_1224
    #%%
temp_anom=np.union1d(anoms['scores'],anoms['scores_V'])
maxs=np.union1d(anoms['maxvar'],anoms['maxmaxmin'])
temp_anom=np.setdiff1d(temp_anom,maxs)
#temp_anom=np.setdiff1d(temp_anom,anoms['maxmaxmin'])
temp_anom.shape
#%%
temp_anom=np.union1d(anoms32['scores'],anoms32['scores_V'])
maxs=np.union1d(anoms32['maxvar'],anoms32['maxmaxmin'])
tt=np.setdiff1d(maxs,temp_anom)
s=np.setdiff1d(temp_anom,maxs)
total=np.union1d(temp_anom,maxs)
backtoback=[]
for i in tt:
    if np.min(np.abs(temp_anom- i)) < 3:
        backtoback.append(i)
print(len(backtoback))
tt=np.setdiff1d(tt,backtoback)
print(tt.shape)
#%%
def rep_check(inp):
    output=[]
    for i in range(inp.shape[0]-1):
        if not np.min(np.abs(inp[i+1]- inp[i])) < 3:
            output.append(inp[i])
    output=np.array(output)
    return output
#%%
riz=np.setdiff1d(rep_check(anoms3['maxvar']),rep_check(anoms31['maxvar']))
#%%
for anom in np.arange(145,166):
    print(anom)
    
    plt.subplot(221)
    for i in [0,1,2]:
        plt.plot(select_1224[i][anom*int(SampleNum/2)-80:(anom*int(SampleNum/2)+80)])
    plt.legend('A' 'B' 'C')
    plt.title('V')
        
    plt.subplot(222)
    for i in [3,4,5]:
        plt.plot(select_1224[i][anom*int(SampleNum/2)-80:(anom*int(SampleNum/2)+80)])
    plt.legend('A' 'B' 'C')
    plt.title('I')  
    
    plt.subplot(223)
    for i in [6,7,8]:
        plt.plot(select_1224[i][anom*int(SampleNum/2)-80:(anom*int(SampleNum/2)+80)])
    plt.legend('A' 'B' 'C') 
    plt.title('P')    
    
    plt.subplot(224)
    for i in [9,10,11]:
        plt.plot(select_1224[i][anom*int(SampleNum/2)-80:(anom*int(SampleNum/2)+80)])
    plt.legend('A' 'B' 'C')
    plt.title('Q')    

    plt.show()

 #%%
plt.show()
plt.subplot(221)
data=(whole_features['scores_V']+1).reshape(-1,1)
#data=np.log(data)
xt, lmbda = stats.boxcox((whole_features['scores_V'])+1)
plt.hist(xt, bins=1000, density=True, alpha=0.6, color='g')
#plt.xlim(-4, -0.5)
#plt.ylim(0, 0.03)
#plt.axis('off')
plt.title('GAN_V')
#plt.xlabel('a')
plt.gca().axes.get_xaxis().set_ticklabels([])


plt.subplot(222)
data=(whole_features['scores_scale']+1).reshape(-1,1)
#data=np.log(data)
xt, lmbda = stats.boxcox((whole_features['scores'])+1)
plt.hist(data, bins=1000, density=True, alpha=0.6, color='g')
plt.xlim(-1.5, 2.5)
plt.title('GAN')
#plt.xlabel('b')
plt.gca().axes.get_xaxis().set_ticklabels([])


plt.subplot(223)
data=(whole_features['maxmaxmin']).reshape(-1,1)
data=np.log(data)
plt.xlim(-3, -1)
plt.hist(data, bins=1000, density=True, alpha=0.6, color='g')
plt.title('maxmin')
#plt.xlabel('c')
plt.gca().axes.get_xaxis().set_ticklabels([])
#plt.gca().axes.get_xaxis().set_visible(False)

plt.subplot(224)
data=(whole_features['maxvar']).reshape(-1,1)
data=np.log(data)
plt.xlim(-10, -5)
plt.hist(data, bins=1000, density=True, alpha=0.6, color='g')
plt.title('maxvar')
#plt.xlabel('d')

plt.gca().axes.get_xaxis().set_ticklabels([])

#plt.savefig('figures/paper/before_transformation.pdf')
plt.show()
#%%
plt.show()
plt.subplot(221)
data=(whole_features['scores_scale_V']).reshape(-1,1)
data=np.log(data)
plt.hist(data, bins=1000, density=True, alpha=0.6, color='g')
plt.xlim(-4, -0.5)
#plt.ylim(0, 0.03)
#plt.axis('off')
plt.title('GAN_V')
#plt.xlabel('a')
plt.gca().axes.get_xaxis().set_ticklabels([])


plt.subplot(222)
data=(whole_features['scores_scale']).reshape(-1,1)
#data=np.log(data)
plt.hist(data, bins=1000, density=True, alpha=0.6, color='g')
plt.xlim(-2, 2)
plt.title('GAN')
#plt.xlabel('b')
plt.gca().axes.get_xaxis().set_ticklabels([])


plt.subplot(223)
data=(whole_features['maxmaxmin']).reshape(-1,1)
data=np.log(data)
#data=np.log(data)
n=2
clf = mixture.GaussianMixture(n_components=n, covariance_type='full')
clf.fit(data)

lamb=3
data=(data-np.mean(clf.means_))**lamb/lamb
plt.xlim(-0.1, 0.1)
plt.hist(data, bins=10000, density=True, alpha=0.6, color='g')
plt.title('maxmin')
#plt.xlabel('c')
plt.gca().axes.get_xaxis().set_ticklabels([])
#plt.gca().axes.get_xaxis().set_visible(False)

plt.subplot(224)
data=(whole_features['maxvar']).reshape(-1,1)
data=np.log(data)
n=2
clf = mixture.GaussianMixture(n_components=n, covariance_type='full')
clf.fit(data)
#data=np.log(data)
lamb=3
data=(data-np.mean(clf.means_))**lamb/lamb
plt.xlim(-0.5, 0.75)
plt.hist(data, bins=10000, density=True, alpha=0.6, color='g')
plt.title('maxvar')
#plt.xlabel('d')

plt.gca().axes.get_xaxis().set_ticklabels([])

plt.savefig('figures/paper/after_transformation_GMMmean.pdf')
plt.show()
#%%















# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # # # # # find the main accuracy in the following code
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================




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




#%%
filename='data/Armin_Data/July_03/pkl/J3.pkl'
select_1224=load_real_data(filename)
    #%%
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
# =============================================================================
# #      detected anomalies with zp=3.1
# =============================================================================
# =============================================================================
zp=3.1

names=['maxvar','maxmaxmin']
detected_anoms={}
for i,d in enumerate(data):
    dt = d
    # Fit a normal distribution to the data:
    mu, std = norm.fit(dt)
    
    high=mu+zp*std
    low=mu-zp*std
    anoms_1224=np.union1d(np.where(dt>=high)[0], np.where(dt<=low)[0])
    print(anoms_1224.shape)
    detected_anoms[names[i]]=anoms_1224
#%%
# =============================================================================
# =============================================================================
# # uninon of basic mdoel anoms
# =============================================================================
# =============================================================================
basic_union=np.array([])
for f in basic_anoms:
    basic_union=np.union1d(basic_anoms[f],basic_union)
basic_union_unique=rep_check(basic_union)
#%%
# =============================================================================
# =============================================================================
# # uninon of detected mdoel anoms
# =============================================================================
# =============================================================================
detected_union=np.array([])
for f in detected_anoms:
    detected_union=np.union1d(detected_anoms[f],detected_union)
detected_union_unique=rep_check(detected_union)

#%%
# =============================================================================
# =============================================================================
# =============================================================================
# # # different of basic and detected
# =============================================================================
# =============================================================================
# =============================================================================
diff_basic_detected=np.setdiff1d(basic_union,detected_union)
diff_basic_detected_unique=rep_check(diff_basic_detected)
#%%
dst='figures/all_events/July_03/acc/diff'
for anom in diff_basic_detected_unique:
        print(anom)
        anom=int(anom)
        plt.subplot(221)
        for i in [0,1,2]:
            plt.plot(select_1224[i][anom*int(SampleNum/2)-120:(anom*int(SampleNum/2)+120)])
        plt.legend('A' 'B' 'C')
        plt.title('V')
            
        plt.subplot(222)
        for i in [3,4,5]:
            plt.plot(select_1224[i][anom*int(SampleNum/2)-120:(anom*int(SampleNum/2)+120)])
        plt.legend('A' 'B' 'C')
        plt.title('I')  
        
        plt.subplot(223)
        for i in [6,7,8]:
            plt.plot(select_1224[i][anom*int(SampleNum/2)-120:(anom*int(SampleNum/2)+120)])
        plt.legend('A' 'B' 'C') 
        plt.title('P')    
        
        plt.subplot(224)
        for i in [9,10,11]:
            plt.plot(select_1224[i][anom*int(SampleNum/2)-120:(anom*int(SampleNum/2)+120)])
        plt.legend('A' 'B' 'C')
        plt.title('Q')    
        
        figname=dst+"/"+str(anom)
        plt.savefig(figname)
        plt.show()
        #%%%
        
        
# =============================================================================
# =============================================================================
# # save detected events
# =============================================================================
# =============================================================================
dst='figures/all_events/July_03/acc/detected'
for anom in detected_union_unique:
        print(anom)
        anom=int(anom)
        plt.subplot(221)
        for i in [0,1,2]:
            plt.plot(select_1224[i][anom*int(SampleNum/2)-120:(anom*int(SampleNum/2)+120)])
        plt.legend('A' 'B' 'C')
        plt.title('V')
            
        plt.subplot(222)
        for i in [3,4,5]:
            plt.plot(select_1224[i][anom*int(SampleNum/2)-120:(anom*int(SampleNum/2)+120)])
        plt.legend('A' 'B' 'C')
        plt.title('I')  
        
        plt.subplot(223)
        for i in [6,7,8]:
            plt.plot(select_1224[i][anom*int(SampleNum/2)-120:(anom*int(SampleNum/2)+120)])
        plt.legend('A' 'B' 'C') 
        plt.title('P')    
        
        plt.subplot(224)
        for i in [9,10,11]:
            plt.plot(select_1224[i][anom*int(SampleNum/2)-120:(anom*int(SampleNum/2)+120)])
        plt.legend('A' 'B' 'C')
        plt.title('Q')    
        
        figname=dst+"/"+str(anom)
        plt.savefig(figname)
        plt.show()


#%%
# =============================================================================
# =============================================================================
# # scatter plot of just GAN model with two scores as feature 
# =============================================================================
# =============================================================================
import matplotlib
#matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.usetex'] = False
fig, ax = plt.subplots() # create a new figure with a default 111 subplot

ax.scatter(whole_features['scores_scale'],whole_features['scores_scale_V'],c=whole_features['color'],label=whole_features['color'])
#ax.legend(['r','b'], ['event', 'normal'], loc="lower left")

plt.gca().axes.get_xaxis().set_ticklabels([])
plt.gca().axes.get_yaxis().set_ticklabels([])
#'\\textit{Velocity (\N{DEGREE SIGN}/sec)}
plt.xlabel('Score from main GAN_{i,p,q}',fontsize=25)
plt.ylabel('Score from GAN_{v}',fontsize=25)
#plt.label('Normal', 'Event',fontsize=25)

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
axins = zoomed_inset_axes(ax, 8, loc=4) # zoom-factor: 2.5, location: upper-left

axins.scatter(whole_features['scores_scale'],whole_features['scores_scale_V'],c=whole_features['color'])


x1, x2, y1, y2 = -6, 6, -10, 3 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits

plt.yticks(visible=False)
plt.xticks(visible=False)

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")

#plt.savefig('figures\paper\GAN_GANV.png',dpi=10000)
#plt.show()
#%%


# =============================================================================
# =============================================================================
# # plot sum 
# =============================================================================
# =============================================================================


#%%
temp_anom=np.union1d(anoms['scores'],anoms['scores_V'])
maxs=np.union1d(anoms['maxvar'],anoms['maxmaxmin'])
temp_anom=np.setdiff1d(temp_anom,maxs)
#temp_anom=np.setdiff1d(temp_anom,anoms['maxmaxmin'])
temp_anom.shape
#%%
temp_anom=np.union1d(anoms32['scores'],anoms32['scores_V'])
maxs=np.union1d(anoms32['maxvar'],anoms32['maxmaxmin'])
tt=np.setdiff1d(maxs,temp_anom)
s=np.setdiff1d(temp_anom,maxs)
total=np.union1d(temp_anom,maxs)
backtoback=[]
for i in tt:
    if np.min(np.abs(temp_anom- i)) < 3:
        backtoback.append(i)
print(len(backtoback))
tt=np.setdiff1d(tt,backtoback)
print(tt.shape)
#%%
def rep_check(inp):
    output=[]
    for i in range(inp.shape[0]-1):
        if not np.min(np.abs(inp[i+1]- inp[i])) < 3:
            output.append(inp[i])
    output=np.array(output)
    return output
#%%
riz=np.setdiff1d(rep_check(anoms3['maxvar']),rep_check(anoms31['maxvar']))

