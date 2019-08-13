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
            whole_data=np.append(whole_data,selected_data.values,axis=0)
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
        
    
    #dir_name="data/Armin_Data/July_03/pkl"
    #os.mkdir(dir_name)
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

# ===================