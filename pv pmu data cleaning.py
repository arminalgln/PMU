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
filenames=os.listdir("data")
#%%
# importing data from a file function
def OneFileImport(filename):
    dir_name="data"
    base_filename=filename
    path=os.path.join(dir_name, base_filename)
    imported_data=pd.read_csv(path,header=None)
    return imported_data
    
#%%
data=OneFileImport(filenames[0])
data=data[[1,3,4,5]]
data=data.rename(index=str, columns={1: "flag", 3: "date",4:"time",5:"value"})
#%%
groups=data.groupby('flag')
#%%
flags=data['flag'].unique()[0:13]
minn=1000000
for f in flags:
    g=groups.get_group(f)
   
    if g.shape[0]<=minn:
        minn=g.shape[0]
        
print(minn)

selected_data={}
for f in flags:
    selected_data[f]=groups.get_group(f).value.values.astype(float)[0:minn]
    
selected_data['time']=groups.get_group(f).time.values.astype(float)[0:minn]

selected_data=pd.DataFrame(selected_data)

selected_data=selected_data.drop('UCR_PSL_UPMU:QF',axis=1)

#%%

features=['L1MAG','L2MAG', 'L3MAG','C1MAG',
       'C2MAG', 'C3MAG', 'PA', 'PB', 'PC', 'QA', 'QB', 'QC']
    

selected_data=selected_data.rename(index=str, columns={'_PSL_UPMU-PM1:V':'L1MAG', '_PSL_UPMU-PA1:VH':'L1ANG', '_PSL_UPMU-PM2:V':'L2MAG',
       '_PSL_UPMU-PA2:VH':'L2ANG', '_PSL_UPMU-PM3:V':'L3MAG', '_PSL_UPMU-PA3:VH':'L3ANG',
       '_PSL_UPMU-PM4:I':'C1MAG', '_PSL_UPMU-PA4:IH':'C1ANG', '_PSL_UPMU-PM5:I':'C2MAG',
       '_PSL_UPMU-PA5:IH':'C2ANG', '_PSL_UPMU-PM6:I':'C3MAG', '_PSL_UPMU-PA6:IH':'C3ANG'})

    
  #%%  
Active={}
Reacive={}
#keys={}
pf={}


Active['A']=selected_data['L1MAG']*selected_data['C1MAG']*(np.cos((selected_data['L1ANG']-selected_data['C1ANG'])*(np.pi/180)))
Active['B']=selected_data['L2MAG']*selected_data['C2MAG']*(np.cos((selected_data['L2ANG']-selected_data['C2ANG'])*(np.pi/180)))
Active['C']=selected_data['L3MAG']*selected_data['C3MAG']*(np.cos((selected_data['L3ANG']-selected_data['C3ANG'])*(np.pi/180)))
    
Reacive['A']=selected_data['L1MAG']*selected_data['C1MAG']*(np.sin((selected_data['L1ANG']-selected_data['C1ANG'])*(np.pi/180)))
Reacive['B']=selected_data['L2MAG']*selected_data['C2MAG']*(np.sin((selected_data['L2ANG']-selected_data['C2ANG'])*(np.pi/180)))
Reacive['C']=selected_data['L3MAG']*selected_data['C3MAG']*(np.sin((selected_data['L3ANG']-selected_data['C3ANG'])*(np.pi/180)))
   
pf['A']=Active['A']/np.sqrt(np.square(Active['A'])+np.square(Reacive['A']))
pf['B']=Active['B']/np.sqrt(np.square(Active['B'])+np.square(Reacive['B']))
pf['C']=Active['C']/np.sqrt(np.square(Active['C'])+np.square(Reacive['C']))


selected_data['PA']=Active['A']
selected_data['PB']=Active['B']
selected_data['PC']=Active['C']

selected_data['QA']=Reacive['A']
selected_data['QB']=Reacive['B']
selected_data['QC']=Reacive['C'] 
  
selected_data['pfA']=pf['A']
selected_data['pfB']=pf['B']
selected_data['pfC']=pf['C']
    
#%%
output = open('15minPVPMU.pkl', 'wb')
pickle.dump(selected_data, output)
output.close()




