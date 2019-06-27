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
SampleNum=80
	
start=16626
end=start+SampleNum
power=selected_data['1224']['C1MAG'][start:end]
#items=np.where(y_pred_train[start:end]== -1)
#marks= list(items[0])
plt.plot(power , color='#0033cc',linestyle='-', linewidth=1,label='Actuals')
#plt.plot(power, color='#ff5960',
#         markevery=marks, marker='^', linestyle=' ', markersize=15)

plt.show()
         
SampleNum=80

start=17762
end=start+SampleNum
power=selected_data['1224']['C1MAG'][start:end]
#items=np.where(y_pred_train[start:end]== -1)
#marks= list(items[0])
plt.plot(power , color='#0033cc',linestyle='-', linewidth=1,label='Actuals')
plt.show()
#%%
# =============================================================================
# Different plots

# =============================================================================
plt.plot(selected_data['1224']['C1MAG'][0:100],selected_data['1224']['L1MAG'][0:100])
plt.scatter(selected_data['1224']['C1MAG'][0:100],selected_data['1224']['L1MAG'][0:100])
plt.scatter(selected_data['1224']['C1MAG'],selected_data['1224']['L1MAG'])
plt.scatter(selected_data['1224']['C1MAG']hnjh,selected_data['1224']['L1MAG'])
plt.scatter(selected_data['1224']['C1MAG'][0:1000],selected_data['1224']['L1MAG'][0:1000])
plt.scatter(selected_data['1224']['C1MAG'][0:10000],selected_data['1224']['L1MAG'][0:10000])
plt.scatter(selected_data['1224']['C2MAG'][0:10000],selected_data['1224']['L2MAG'][0:10000])
plt.plot(selected_data['1224']['C2MAG'][0:10000],selected_data['1224']['L2MAG'][0:10000])
plt.plot(selected_data['1224']['PB'][0:10000],selected_data['1224']['PB'][0:10000])
plt.plot(selected_data['1224']['PB'][0:10000],selected_data['1224']['QB'][0:10000])
plt.scatter(selected_data['1224']['PB'][0:10000],selected_data['1224']['QB'][0:10000])
plt.plot(selected_data['1224']['PB'][0:100000],selected_data['1224']['QB'][0:100000])
plt.plot(selected_data['1224']['PB'][0:10000],selected_data['1224']['PB'][
plt.plot(selected_data['1224']['C2MAG'][0:100000],selected_data['1224']['L2MAG'][0:100000])
plt.plot(selected_data['1224']['C2MAG'][0:1000000],selected_data['1224']['L2MAG'][0:1000000])
plt.plot(selected_data['1224']['PB'][0:1000000],selected_data['1224']['QB'][0:1000000])
plt.plot(selected_data['1224']['PB'][0:1000000],np.abs(selected_data['1224']['QB'][0:1000000]))
plt.scatter(selected_data['1224']['C1MAG'][0:100],selected_data['1224']['L1MAG'][0:100])
plt.plot(selected_data['1224']['C1MAG'][0:100],selected_data['1224']['L1MAG'][0:100])
plt.plot(selected_data['1224']['C1MAG'][0:100])
plt.plot(selected_data['1224']['L1MAG'][0:100])
plt.plot(selected_data['1224']['L1MAG'][0:1000])
plt.plot(selected_data['1224']['C1MAG'][0:1000])
plt.plot(selected_data['1224']['C1MAG'][600:800])
plt.plot(selected_data['1224']['L1MAG'][600:800])
plt.plot(selected_data['1224']['C1MAG'][600:800],selected_data['1224']['L1MAG'][600:800])


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