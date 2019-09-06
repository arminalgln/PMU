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
import natsort
from scipy.io import loadmat
from math import ceil
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
# Pythono3 code to rename multiple 
# files in a directory or folder 

# =============================================================================
# =============================================================================
# =============================================================================
# # # Reanme the file names in a folder
# =============================================================================
# =============================================================================
# =============================================================================
for n in np.arange(4,18):
    if n<10:
        dir="data/Armin_Data/July_0"+str(n)+"/"
    else:
        dir="data/Armin_Data/July_"+str(n)+"/"
    
    # Function to rename multiple files 
    def main(): 
    	i = 0
    	
    	for filename in os.listdir(dir)[24:48]: 
    		dst =str(i) + ".csv"
    		src =dir+ filename 
    		dst =dir+ dst 
    		
    		# rename() function will 
    		# rename all the files 
    		os.rename(src, dst) 
    		i += 1
    
    # Driver Code 
    if __name__ == '__main__': 
    	
    	# Calling main() function 
    	main() 
       
        # whole data filenames in the data directory
    if n<10:
        dir="data/Armin_Data/July_0"+str(n)
    else:
        dir="data/Armin_Data/July_"+str(n)
    foldernames=os.listdir(dir)
    filenames1224=foldernames[0:24]
    #filenames1224.sort(key=lambda f: int(filter(str.isdigit, f)))
    filenames1224=natsort.natsorted(filenames1224)
    
    #active and reactive power consumption calculation
    whole_data=[]
    #filenames1224.sort(key=lambda f: int(filter(str.isdigit, f)))
    for count,i in enumerate(filenames1224):
        
        Active={}
        Reacive={}
        keys={}
        pf={}
        
        selected_data=OneFileImport(i,dir)    
        
        Active['A']=selected_data['L1Mag']*selected_data['C1Mag']*(np.cos((selected_data['L1Ang']-selected_data['C1Ang'])*(np.pi/180)))
        Active['B']=selected_data['L2Mag']*selected_data['C2Mag']*(np.cos((selected_data['L2Ang']-selected_data['C2Ang'])*(np.pi/180)))
        Active['C']=selected_data['L3Mag']*selected_data['C3Mag']*(np.cos((selected_data['L3Ang']-selected_data['C3Ang'])*(np.pi/180)))
            
        Reacive['A']=selected_data['L1Mag']*selected_data['C1Mag']*(np.sin((selected_data['L1Ang']-selected_data['C1Ang'])*(np.pi/180)))
        Reacive['B']=selected_data['L2Mag']*selected_data['C2Mag']*(np.sin((selected_data['L2Ang']-selected_data['C2Ang'])*(np.pi/180)))
        Reacive['C']=selected_data['L3Mag']*selected_data['C3Mag']*(np.sin((selected_data['L3Ang']-selected_data['C3Ang'])*(np.pi/180)))
        #   
        #pf['A']=Active['A']/np.sqrt(np.square(Active['A'])+np.square(Reacive['A']))
        #pf['B']=Active['B']/np.sqrt(np.square(Active['B'])+np.square(Reacive['B']))
        #pf['C']=Active['C']/np.sqrt(np.square(Active['C'])+np.square(Reacive['C']))
        
        
        selected_data['PA']=Active['A']
        selected_data['PB']=Active['B']
        selected_data['PC']=Active['C']
        
        selected_data['QA']=Reacive['A']
        selected_data['QB']=Reacive['B']
        selected_data['QC']=Reacive['C'] 
        
        selected_data=selected_data.drop(columns=['Unnamed: 0','L1Ang','L2Ang','L3Ang','C1Ang','C2Ang','C3Ang'])
        
        if count==0:
            whole_data=selected_data.values
        else:
            whole_data=np.(whole_data,selected_data.values,axis=0)
    #    whole_data.append(selected_data.values,axis=0)
        print(i)
    
    k=['L1MAG','L2MAG', 'L3MAG','C1MAG',
           'C2MAG', 'C3MAG', 'PA', 'PB', 'PC', 'QA', 'QB', 'QC']
        
    day_data={}
    day_data['1224']={}
    c=0
    for key in k:
        day_data['1224'][key]=whole_data[:,c]
        c+=1
        
    if n<10:
        dir="data/Armin_Data/July_0"+str(n)+"/pkl"
    else:
        dir="data/Armin_Data/July_"+str(n)+"/pkl"
    dir_name=dir
    os.mkdir(dir_name)
        # write python dict to a file
    if n<10:
        dir="data/Armin_Data/July_0"+str(n)+"/pkl/jul" + str(n) + ".pkl"
    else:
        dir="data/Armin_Data/July_"+str(n)+"/pkl/jul" + str(n) + ".pkl"
    output = open(dir, 'wb')
    pickle.dump(day_data, output)
    output.close()
    print(n)
#%%
     #read a pickle file
    pkl_file = open('CompleteOneDay.pkl', 'rb')
    selected_data = pickle.load(pkl_file)
    pkl_file.close()
    print(n)
#%%
# =============================================================================
# =============================================================================
# # 
# # find new pointer for july 03 from alireza new time file sent by email sep 3 2019
# # =============================================================================
# 
# =============================================================================
time_file='data/Armin_Data/July_03/'
new_time = loadmat(time_file+'time.mat')['time']
new_time=new_time.ravel()
#%%
# =============================================================================
# =============================================================================
# # vectorize the ceiling function
# =============================================================================
# =============================================================================
def f(x):
    return np.ceil(x)
ceil2 = np.vectorize(f)
new_time=ceil2(new_time/100000)
#%%
times=np.array([])
for hour in range(24):
    temp_times=pd.read_csv(time_file+str(hour)+'.csv')['Unnamed: 0']
    times=np.concatenate((times,temp_times))
#%%
times=np.array(times)
#%%
times=times.ravel()
#%%
times=ceil2(times/100000)
#%%
#new_pointer=[]
#s=times.shape
#for point in range(s):
#    if times[point] in new_time:
#        new_pointer.append(point)
#    else:
#        print(point)
    #%%

diff=np.setdiff1d(times,new_time)
uni=np.union1d(times,new_time)
inter=np.intersect1d(times,new_time)
        
#%%
records_array = times
idx_sort = np.argsort(records_array)
sorted_records_array = records_array[idx_sort]
vals, idx_start, count = np.unique(sorted_records_array, return_counts=True,
                                return_index=True)

# sets of indices
res = np.split(idx_sort, idx_start[1:])
#filter them with respect to their size, keeping only items occurring more than once

vals = vals[count > 1]
res = filter(lambda x: x.size > 1, res)
        
        
    
    
#%%
# =============================================================================
# =============================================================================
# # time list for 1200
# =============================================================================
# =============================================================================

times_1200=np.array([])
for hour in range(24):
    temp_times=pd.read_csv(time_file+'Bld_1200_'+str(hour+1)+'.csv')['Unnamed: 0']
    times_1200=np.concatenate((times_1200,temp_times))
#%%
times_1200=np.array(times_1200)
#%%
times_1200=times_1200.ravel()
#%%
times_1200=ceil2(times_1200/100000)

#%%
diff=np.setdiff1d(times_1200,new_time)
uni=np.union1d(times_1200,new_time)
inter=np.intersect1d(times_1200,new_time)
        
#%%
records_array = times_1200
idx_sort = np.argsort(records_array)
sorted_records_array = records_array[idx_sort]
vals, idx_start, count = np.unique(sorted_records_array, return_counts=True,
                                return_index=True)

# sets of indices
res = np.split(idx_sort, idx_start[1:])
#filter them with respect to their size, keeping only items occurring more than once

vals = vals[count > 1]
res = filter(lambda x: x.size > 1, res)
#%%
old_pointer=loadmat('data/pointer.mat')['pointer']['Jul_03'][0].ravel()[0].ravel()
 
new_pointer=np.array([])     
for point in old_pointer:
    tempt=times_1200[point]
    
    p=np.where(new_time==tempt)
    print(p)
    new_pointer=np.append(new_pointer,p)
    
#%%
# =============================================================================
# =============================================================================
# #     use the new pointer to extract the anomalies in the main data from alirezas method
# =============================================================================
# =============================================================================
    
    
# =============================================================================
#     load real data
# =============================================================================
filename='data/Armin_Data/July_03/pkl/J3.pkl'
def load_real_data(filename):
         #read a pickle file
         
    pmu='1224'

    pkl_file = open(filename, 'rb')
    selected_data = pkl.load(pkl_file)
    pkl_file.close()
    selected_data=pd.DataFrame(selected_data)
    selected_data=selected_data.fillna(method='ffill')
    print(selected_data.keys())
    data=selected_data[pmu]
    features=['L1MAG','L2MAG', 'L3MAG','C1MAG',
       'C2MAG', 'C3MAG', 'PA', 'PB', 'PC', 'QA', 'QB', 'QC']
    
    select=[]
    for f in features:
        select.append(list(data[f]))
    
    select=np.array(select)
    
    
    return select
#%%
select_1224=load_real_data(filename)
    #%%
new_pointer.sort()


dst="figures/1224_15_days/July_03/window"


# =============================================================================
#     save the window method event points
# =============================================================================

for anom in old_pointer:
    anom=int(anom)
    print(anom)
    
    plt.subplot(221)
    for i in [0,1,2]:
        plt.plot(select_1224[i][anom-120:(anom+240)])
    plt.legend('A' 'B' 'C')
    plt.title('V')
        
    plt.subplot(222)
    for i in [3,4,5]:
        plt.plot(select_1224[i][anom-120:(anom+240)])
    plt.legend('A' 'B' 'C')
    plt.title('I')  
    
    plt.subplot(223)
    for i in [6,7,8]:
        plt.plot(select_1224[i][anom-120:(anom+240)])
    plt.legend('A' 'B' 'C') 
    plt.title('P')    
    
    plt.subplot(224)
    for i in [9,10,11]:
        plt.plot(select_1224[i][anom-120:(anom+240)])
    plt.legend('A' 'B' 'C')
    plt.title('Q')    
    figname=dst+"/"+str(anom)
    plt.savefig(figname)
    plt.show()






#%%%%
files='data/Armin_Data/July_03/Hunter_1224_'
v1=np.array([])
for hour in range(24):
    print(hour)
    v1temp=pd.read_csv(files+str(hour+1)+'.csv')['L1Mag']
    v1=np.concatenate((v1,v1temp),axis=None)
    
plt.plot(v1)
#%%
# importing data from a file function
def OneFileImport(filename,dir):
    dir_name=dir
    base_filename=filename
    path=os.path.join(dir_name, base_filename)
    imported_data=pd.read_csv(path)
    return imported_data
    
#%%
for n in [3]:
    if n<10:
        num='0'+str(n)
    else:
        num=str(n)
    
    dir='data/Armin_Data/July_'+num
    
    foldernames=os.listdir(dir)
    selected_files=np.array([])
    for f in fl:
        spl=f.split('_')
        if 'Hunter' in spl:
            selected_files=np.append(selected_files,f)
#    filenames1224=foldernames[0:24]
    #filenames1224.sort(key=lambda f: int(filter(str.isdigit, f)))
    filenames1224=natsort.natsorted(selected_files)
    
    #active and reactive power consumption calculation
    whole_data=np.array([])
    #filenames1224.sort(key=lambda f: int(filter(str.isdigit, f)))
    for count,file in enumerate(filenames1224):
        print(count,file)
        Active={}
        Reacive={}
        keys={}
        pf={}
        
        selected_data=OneFileImport(file,dir)    
        
        Active['A']=selected_data['L1Mag']*selected_data['C1Mag']*(np.cos((selected_data['L1Ang']-selected_data['C1Ang'])*(np.pi/180)))
        Active['B']=selected_data['L2Mag']*selected_data['C2Mag']*(np.cos((selected_data['L2Ang']-selected_data['C2Ang'])*(np.pi/180)))
        Active['C']=selected_data['L3Mag']*selected_data['C3Mag']*(np.cos((selected_data['L3Ang']-selected_data['C3Ang'])*(np.pi/180)))
            
        Reacive['A']=selected_data['L1Mag']*selected_data['C1Mag']*(np.sin((selected_data['L1Ang']-selected_data['C1Ang'])*(np.pi/180)))
        Reacive['B']=selected_data['L2Mag']*selected_data['C2Mag']*(np.sin((selected_data['L2Ang']-selected_data['C2Ang'])*(np.pi/180)))
        Reacive['C']=selected_data['L3Mag']*selected_data['C3Mag']*(np.sin((selected_data['L3Ang']-selected_data['C3Ang'])*(np.pi/180)))
        #   
        #pf['A']=Active['A']/np.sqrt(np.square(Active['A'])+np.square(Reacive['A']))
        #pf['B']=Active['B']/np.sqrt(np.square(Active['B'])+np.square(Reacive['B']))
        #pf['C']=Active['C']/np.sqrt(np.square(Active['C'])+np.square(Reacive['C']))
        
        
        selected_data['PA']=Active['A']
        selected_data['PB']=Active['B']
        selected_data['PC']=Active['C']
        
        selected_data['QA']=Reacive['A']
        selected_data['QB']=Reacive['B']
        selected_data['QC']=Reacive['C'] 
        
        selected_data=selected_data.drop(columns=['Unnamed: 0','L1Ang','L2Ang','L3Ang','C1Ang','C2Ang','C3Ang'])
        
        if count==0:
            whole_data=selected_data.values
        else:
            whole_data=np.append(whole_data,selected_data.values,axis=0)
    #    whole_data.append(selected_data.values,axis=0)
#        print(i)
    
    k=['L1MAG','L2MAG', 'L3MAG','C1MAG',
           'C2MAG', 'C3MAG', 'PA', 'PB', 'PC', 'QA', 'QB', 'QC']
        
    day_data={}
    day_data['1224']={}
    c=0
    for key in k:
        day_data['1224'][key]=whole_data[:,c]
        c+=1

    dir=dir+'/pkl/J'+str(n)+'.pkl'
        # write python dict to a file
    
    output = open(dir, 'wb')
    pickle.dump(day_data, output)
    output.close()
    print(n)
