
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:26:15 2019

@author: hamed
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import os
import pickle as pkl
import matplotlib.pyplot as plt
import operator
import math
#%%
# =============================================================================
# =============================================================================
# # read one file of the PMU data , each file is for 10 minutes 
# =============================================================================
# =============================================================================

# whole data filenames in the data directory
filenames=os.listdir("data/jul 1")
#%%
# importing data from a file function
def cleancsv(filename):
    dir_name="data/jul 1"
    base_filename=filename
    pathr=os.path.join(dir_name, base_filename)
#    imported_data=pd.read_csv(path,header=None, error_bad_lines=False)
    with open(pathr,'r') as f:
        dir_name="data/jul1sorted"
        pathw=os.path.join(dir_name,filename)
        with open(pathw,'w') as f1:
            for i in range(6):
                next(f) # skip header line
            for line in f:
                f1.write(line)

#%%
for file in filenames:
    cleancsv(file)
#%%
# =============================================================================
# =============================================================================
# #     make time 
# =============================================================================
# =============================================================================
    
samplingrate=60
timenum=3600*samplingrate
timeslots=np.arange(0,timenum).transpose()
    #%%
filenames=os.listdir("data/jul1sorted")
#%%
# importing data from a file function
def OneFileImport(filename):
    dir_name="data/jul1sorted"
    base_filename=filename
    path=os.path.join(dir_name, base_filename)
    imported_data=pd.read_csv(path)
    return imported_data
    
#%%
samplingrate=60
timenum=3600*samplingrate
timeslots=np.arange(0,timenum).transpose()
for file in filenames:
    print(file)
    data=OneFileImport(file)
    k=data.keys()
    data=data.drop(columns=[k[0],k[1],k[2],k[15],k[16]])
    k=data.keys()
    f=['L1MAG','L1ANG','L2MAG','L2ANG','L3MAG','L3ANG','C1MAG','C1ANG','C2MAG','C2ANG','C3MAG','C3ANG']
    for count,i in  enumerate(k):
        data=data.rename(index=str, columns={i:f[count]})
        
    data=data.iloc[0:timenum]
    Active={}
    Reacive={}
    #keys={}
#    pf={}
    selected_data={}
##    
    
    Active['A']=data['L1MAG']*data['C1MAG']*(np.cos((data['L1ANG']-data['C1ANG'])*(np.pi/180)))
    Active['B']=data['L2MAG']*data['C2MAG']*(np.cos((data['L2ANG']-data['C2ANG'])*(np.pi/180)))
    Active['C']=data['L3MAG']*data['C3MAG']*(np.cos((data['L3ANG']-data['C3ANG'])*(np.pi/180)))
        
    Reacive['A']=data['L1MAG']*data['C1MAG']*(np.sin((data['L1ANG']-data['C1ANG'])*(np.pi/180)))
    Reacive['B']=data['L2MAG']*data['C2MAG']*(np.sin((data['L2ANG']-data['C2ANG'])*(np.pi/180)))
    Reacive['C']=data['L3MAG']*data['C3MAG']*(np.sin((data['L3ANG']-data['C3ANG'])*(np.pi/180)))
       
#    pf['A']=Active['A']/np.sqrt(np.square(Active['A'])+np.square(Reacive['A']))
#    pf['B']=Active['B']/np.sqrt(np.square(Active['B'])+np.square(Reacive['B']))
#    pf['C']=Active['C']/np.sqrt(np.square(Active['C'])+np.square(Reacive['C']))
#    
#    
    selected_data['PA']=Active['A']
    selected_data['PB']=Active['B']
    selected_data['PC']=Active['C']
    
    selected_data['QA']=Reacive['A']
    selected_data['QB']=Reacive['B']
    selected_data['QC']=Reacive['C'] 
    features=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG']
    for f in features:
        selected_data[f]=data[f]
        
    selected_data['timeslot']=timeslots
    selected_data['hour']=np.ones(timenum)*(int(file.split(sep='.')[0]))
      
#    selected_data['pfA']=pf['A']
#    selected_data['pfB']=pf['B']
#    selected_data['pfC']=pf['C']
     
    form='.pkl'
    filename=file.split(sep='.')[0]+form
    

    dir_name="data/jul1pkl"
    path=os.path.join(dir_name,filename)
    
    print(path)
    output = open(path, 'wb')
    pkl.dump(selected_data, output)
    output.close()
    
    #%%
    
dirname="data/jul1pkl/1.pkl"
pkl_file = open(dirname, 'rb')
dd=pkl.load(pkl_file)
pkl_file.close()




