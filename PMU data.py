# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:26:15 2019

@author: hamed
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
#%%
# =============================================================================
# =============================================================================
# # read one file of the PMU data , each file is for 10 minutes 
# =============================================================================
# =============================================================================

# whole data filenames in the data directory
filenames=os.listdir("Raw_data")
#%%
# importing data from a file function
def OneFileImport(filename):
    dir_name="Raw_data"
    base_filename=filename
    path=os.path.join(dir_name, base_filename)
    imported_data=pd.read_csv(path)
    return imported_data
    
#%%
data=OneFileImport(filenames[0])
#%%
#pmu locations
SeparateData={}
Locations=['1086','1224','1225','1200']
for loc in Locations:
    SeparateData[loc]={}
columns=data.keys()
Tiemslots=data['Timestamp (ns)'].values
Dates=data['Human-Readable Time (UTC)'].values
for key in columns:
    col=key.split('/')
    if len(col)>1: #to ignore teh time and date
        loc=col[1]
#        print(loc,col)
        entry, index = col[2].split(' ')
#        print(entry)
    if (entry !='LSTATE') and (index=='(Mean)'):
        SeparateData[loc][entry]=data[key]
#%%
SeparateData={}
Locations=['1086','1224','1225','1200']
for loc in Locations:
    SeparateData[loc]={}
Tiemslots=[]
Dates=[] 
triger=0
filecount=0
for file in filenames:
    CollectedData=OneFileImport(file)
    if triger==0:
        Tiemslots=CollectedData['Timestamp (ns)']
        Dates=CollectedData['Human-Readable Time (UTC)']
    if triger==1:
        Tiemslots=np.append(Tiemslots,CollectedData['Timestamp (ns)'])
        Dates=np.append(Dates,CollectedData['Human-Readable Time (UTC)'])
    
    columns=CollectedData.keys()
    for key in columns:
        col=key.split('/')
        if len(col)>1: #to ignore teh time and date
            loc=col[1]
    #        print(loc,col)
            entry, index = col[2].split(' ')
    #        print(entry)
        if (entry !='LSTATE') and (index=='(Mean)'):
            if triger==0:
                SeparateData[loc][entry]=CollectedData[key]
            if triger==1:
                SeparateData[loc][entry]=np.append(SeparateData[loc][entry],CollectedData[key])
#    if filecount==2:
#        break
    triger=1
    print(filecount)
    filecount=filecount+1
    #%%
    # write python dict to a file
outputt = open('OneDay.pkl', 'wb')
pickle.dump(SeparateData, outputt)
outputt.close()
#%%
 #read a pickle file
pkl_file = open('OneDay.pkl', 'rb')
selected_data = pickle.load(pkl_file)
pkl_file.close()

        
        
    
    
    
    
    
    
    
    
    
    
        
        
    