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
N=200
pmu='1224'
shift=int(SampleNum/2)

train_data=[]
for i in range(N):
    train_data.append(selected_data[pmu]['C1MAG'][start+i*shift:end+i*shift]-np.mean(selected_data[pmu]['C1MAG'][start+i*shift:end+i*shift]))


#%%
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(train_data)










