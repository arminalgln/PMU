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
SampleNum=100000
start=0
end=start+SampleNum
power=selected_data['1224']['pfC'][start:end]
items=np.where(y_pred_train[start:end]== -1)
marks= list(items[0])
plt.plot(power , color='#0033cc',linestyle='-', linewidth=1,label='Actuals')
plt.plot(power, color='#ff5960',
         markevery=marks, marker='^', linestyle=' ', markersize=5)
