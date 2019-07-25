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
import operator
import math
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
Locations=['1086','1224','1225','1200']
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
#%%
#active and reactive power consumption calculation

Active={}
Reacive={}
keys={}
pf={}
for loc in Locations:
    k=list(selected_data[loc].keys())    
    keys[loc]=sorted(k)
    Active[loc]={}
    Reacive[loc]={}
    pf[loc]={}
    
for loc in Locations:
    Active[loc]['A']=selected_data[loc]['L1MAG']*selected_data[loc]['C1MAG']*(np.cos((selected_data[loc]['L1ANG']-selected_data[loc]['C1ANG'])*(np.pi/180)))
    Active[loc]['B']=selected_data[loc]['L2MAG']*selected_data[loc]['C2MAG']*(np.cos((selected_data[loc]['L2ANG']-selected_data[loc]['C2ANG'])*(np.pi/180)))
    Active[loc]['C']=selected_data[loc]['L3MAG']*selected_data[loc]['C3MAG']*(np.cos((selected_data[loc]['L3ANG']-selected_data[loc]['C3ANG'])*(np.pi/180)))
        
    Reacive[loc]['A']=selected_data[loc]['L1MAG']*selected_data[loc]['C1MAG']*(np.sin((selected_data[loc]['L1ANG']-selected_data[loc]['C1ANG'])*(np.pi/180)))
    Reacive[loc]['B']=selected_data[loc]['L2MAG']*selected_data[loc]['C2MAG']*(np.sin((selected_data[loc]['L2ANG']-selected_data[loc]['C2ANG'])*(np.pi/180)))
    Reacive[loc]['C']=selected_data[loc]['L3MAG']*selected_data[loc]['C3MAG']*(np.sin((selected_data[loc]['L3ANG']-selected_data[loc]['C3ANG'])*(np.pi/180)))
       
    pf[loc]['A']=Active[loc]['A']/np.sqrt(np.square(Active[loc]['A'])+np.square(Reacive[loc]['A']))
    pf[loc]['B']=Active[loc]['B']/np.sqrt(np.square(Active[loc]['B'])+np.square(Reacive[loc]['B']))
    pf[loc]['C']=Active[loc]['C']/np.sqrt(np.square(Active[loc]['C'])+np.square(Reacive[loc]['C']))
    
    
    selected_data[loc]['PA']=Active[loc]['A']
    selected_data[loc]['PB']=Active[loc]['B']
    selected_data[loc]['PC']=Active[loc]['C']
    
    selected_data[loc]['QA']=Reacive[loc]['A']
    selected_data[loc]['QB']=Reacive[loc]['B']
    selected_data[loc]['QC']=Reacive[loc]['C'] 
  
    selected_data[loc]['pfA']=pf[loc]['A']
    selected_data[loc]['pfB']=pf[loc]['B']
    selected_data[loc]['pfC']=pf[loc]['C']
    
   
#%%
    
    # write python dict to a file
output = open('CompleteOneDay.pkl', 'wb')
pickle.dump(selected_data, output)
output.close()


#%%
 #read a pickle file
pkl_file = open('CompleteOneDay.pkl', 'rb')
selected_data = pickle.load(pkl_file)
pkl_file.close()

#%%

# =============================================================================
# =============================================================================
# # it gets a vector which is a voltage angle of one phase and it will return frequancy diffrence in each time
# =============================================================================
# =============================================================================
def frequency(angle,span):
    
    span=40
    for i in range(int(angle.shape[0]/span)):
    selected_angle=angle[i*span:i*(span)]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    return df
    
#%%
def P2R(r, angles):
    return r * np.exp(1j*angles)

def R2P(x):
    return abs(x), angle(x)
#%%
r=selected_data['L1MAG'][11500:12000]
ang=(selected_data['L1ANG'][11500:12000]+180)*(2*np.pi/180)
v=P2R(r,ang)
p=selected_data['PA'][11500:12000]
vrated=7200
r=r/vrated
#%%
mat=[np.ones(r.shape[0]),r,r**2]

mat=np.array(mat).transpose()
#%%
a=np.linalg.lstsq(mat,p)
coeff=a[0]
#%%
pgen=np.matmul(mat,coeff)
plt.plot(np.absolute(pgen))
plt.plot(np.absolute(list(p.values)))
plt.show()