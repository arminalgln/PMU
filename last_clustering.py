import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
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
import random
import pickle as pkl
import operator
import math
from sklearn import preprocessing
from keras.models import load_model
import time
from scipy.stats import norm
from scipy.io import loadmat
from natsort import natsorted
from scipy import stats
from seaborn import heatmap
import loading_data
from loading_data import load_real_data, load_standardized_data,load_train_data,load_train_data_V,load_train_vitheta_data_V,load_data_with_features,load_standardized_data_with_features
#%%
# =============================================================================
# =============================================================================
# =============================================================================
# # # extract candidate for the clusters  which extraxted by hand from July 03
# =============================================================================
# =============================================================================
# =============================================================================

cluster_folder_name='onedayclusters'
cluster_folder=os.listdir(cluster_folder_name)
separarted_events={}
cl_num=0
for cluster in cluster_folder:
    separarted_events[cl_num]=[]
    events=os.listdir(cluster_folder_name+'/'+cluster)
    for ev in events:
        separarted_events[cl_num].append(int(ev.split('.')[0]))
        
    cl_num+=1
#%%
# =============================================================================
# =============================================================================
## call data which includes V, I and theta (9 features)
# =============================================================================
# =============================================================================
filename='data/Armin_Data/July_03/pkl/julseppf3.pkl'
k=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG','TA', 'TB', 'TC']
#%%
# =============================================================================
# =============================================================================
# # standardized data
# =============================================================================
# =============================================================================
dds=load_standardized_data_with_features(filename,k)
#%%
# =============================================================================
# =============================================================================
# # normal data
# =============================================================================
# =============================================================================
dd=load_data_with_features(filename,k)
#%%
# =============================================================================
# =============================================================================
# # train data
# =============================================================================
# =============================================================================
start,SampleNum,N=(0,40,500000)
filename='data/Armin_Data/July_03/pkl/julseppf3.pkl'
k=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG','TA', 'TB', 'TC']
tt=load_train_vitheta_data_V(start,SampleNum,N,filename,k)
#%%
filename='data/Armin_Data/July_13/pkl/rawdata13.pkl'
k=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG','TA', 'TB', 'TC']
dds13=load_standardized_data_with_features(filename,k)
dd13=load_data_with_features(filename,k)
start,SampleNum,N=(0,40,500000)
#filename='data/Armin_Data/July_03/pkl/julseppf3.pkl'
#k=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG','TA', 'TB', 'TC']
#tt10=load_train_vitheta_data_V(start,SampleNum,N,filename,k)
#%%        
# =============================================================================
# =============================================================================
# # max corr coeff funciton based on each two event
# =============================================================================
# =============================================================================
def ccf(anom1,anom2,data):
# =============================================================================
#     480 time duration for each event
# =============================================================================
    scale=6
    shift=0   
    SampleNum=40
    max_corr=-1
    for i in range(120):
        cr=0
        for j in range(9):
            cr+=np.corrcoef(data[j][anom1*int(SampleNum/2)-40*scale+shift:(anom1*int(SampleNum/2)+40*scale+shift)],
                            np.roll(data[j][anom2*int(SampleNum/2)-40*scale+shift:(anom2*int(SampleNum/2)+40*scale+shift)],i-60))[0,1]            
        cr=cr/9
        if cr>max_corr:
            max_corr=cr
    return max_corr

#%%
#%%        
# =============================================================================
# =============================================================================
# # max corr coeff funciton based on each two event
# =============================================================================
# =============================================================================
def ccfWithRepresentatives(anom1,rep1,data_anom):
# =============================================================================
#     480 time duration for each event
# =============================================================================
    scale=6
    shift=0   
    SampleNum=40
    max_corr=-1
    for i in range(120):
        cr=0
        for j in range(9):
            cr+=np.corrcoef(data_anom[j][anom1*int(SampleNum/2)-40*scale+shift:(anom1*int(SampleNum/2)+40*scale+shift)],
                            np.roll(rep1[j],i-60))[0,1]            
        cr=cr/9
        if cr>max_corr:
            max_corr=cr
    return max_corr

#%%
# =============================================================================
# =============================================================================
# # Training model - extract candidate for each pre selected cluster
# =============================================================================
# =============================================================================
      

def candidate_correlation(cluster_events,data):   
    #select number of events that we want to consider in each group for training
    N=len(cluster_events)
    trh=50
    N=min(N,trh)
    corr=np.zeros((N,N))
    #restricted candidate

    selected_events=np.random.choice(cluster_events, N, replace=False)
    for idx1,anom1 in enumerate(selected_events):
        print(idx1)
#        if idx1% 100==0:
#            print('iter num: %i', idx1)
        tic=time.clock()
        for idx2,anom2 in enumerate(selected_events):
            if idx2>=idx1:
#                if idx2% 100==0:
#                    print('iter num: %i', idx2)
                max_corr=ccf(anom1,anom2,data)
                corr[idx1,idx2]=max_corr
            else:
                corr[idx1,idx2]=corr[idx2,idx1]
        toc = time.clock()
        print(toc-tic)
    
    index=np.argmax(sum(corr))
    
    candid=selected_events[index]
    
    return corr,candid

#%%
# =============================================================================
# =============================================================================
# # calculate candidate of each cluster
# =============================================================================
# =============================================================================
representatives={}
for cl in separarted_events:
    
    cluster_event=separarted_events[cl]
    _,representatives[cl]=candidate_correlation(separarted_events[cl],dds)

#%%
# =============================================================================
# =============================================================================
# # show representatives
# =============================================================================
# =============================================================================
for can in representatives:
    show([representatives[can]],dd)
#%%
# =============================================================================
# =============================================================================
# =============================================================================
# # # test the whole events one by one to see the accuracy of the candidates    
# =============================================================================
# =============================================================================
# =============================================================================
test_event_clusters={}

for cl in separarted_events:
    print(cl)
    temp_cluster_evevnts=separarted_events[cl]
    #check with the representative
    count=0
    for event in temp_cluster_evevnts:
        if count<100:
            print(event)
            nearest_distance=-1
            for can in representatives:
                dist=ccf(event,representatives[can],dds)
                if dist>nearest_distance:
                    nearest_distance=dist
                    closest_candidate=can
            test_event_clusters[event]=[closest_candidate,cl]
            count+=1
        

#%%
# =============================================================================
# =============================================================================
# # calculate the accuracy of the models (building multiclass confusion matrix)
# =============================================================================
# =============================================================================
            
# =============================================================================
# =============================================================================
# #             whole clusters even with the one phase events
# =============================================================================
# =============================================================================
cl_cum=len(separarted_events)
confusion_matrix=np.zeros((cl_num,cl_cum))

for cl in separarted_events:
    print(cl)
    temp_cluster_evevnts=separarted_events[cl]
    #check with the representative
    count=0
    for event in temp_cluster_evevnts:
        if count<100:
            confusion_matrix[test_event_clusters[event][1],test_event_clusters[event][0]]+=1
        count+=1  

acc={}
acc['tp']=[]
acc['fp']=[]
acc['fn']=[]
acc['tn']=[]
for i in range(cl_num):
    acc['tp'].append(confusion_matrix[i,i])
    acc['fp'].append(sum(confusion_matrix[:,i])-confusion_matrix[i,i])
    acc['fn'].append(sum(confusion_matrix[i,:])-confusion_matrix[i,i])
    acc['tn'].append(sum(sum(confusion_matrix[:,:]))-acc['tp'][i]-acc['fp'][i]-acc['fn'][i])

#%%
# =============================================================================
# =============================================================================
# #     total accuracy of clustering model
# =============================================================================
# =============================================================================
total_acccuracy=(sum(acc['tp'])+sum(acc['tn']))/(sum(acc['tp'])+sum(acc['tn'])+sum(acc['fp'])+sum(acc['fn']))
F1=(sum(acc['tp'])+sum(acc['tp']))/(sum(acc['tp'])+sum(acc['tp'])+sum(acc['fp'])+sum(acc['fn']))
#%%
ConfMtr={}
methods=['KNN','Kmed','fuzzy-cmedoids','proposed']
distances=['eu','dtw','soft-dtw','mcc']
cl_num=8
for m in methods:
    ConfMtr[m]={}
    for d in distances:
        ConfMtr[m][d]=np.zeros((cl_num,cl_cum))


#%%

cluster_events_number=[100,35,13,100,13,34,100,54]

a={}
a[1]='40,9,9,9,9,9,9,9,2,18,2,2,2,2,2,2,1,1,7,1,1,1,1,1,9,9,9,40,9,9,9,9,1,1,1,1,7,1,1,1,2,2,2,2,2,18,2,2,9,9,9,9,9,9,40,9,4,4,4,4,4,4,4,26'
a[2]='55,6,6,7,6,6,7,6,2,22,2,2,2,2,2,2,1,1,8,1,1,1,1,1,7,6,6,55,6,6,7,6,1,1,1,1,8,1,1,1,2,2,2,2,2,21,2,2,7,6,6,7,6,6,55,6,3,3,3,3,3,3,3,32'
a[3]='53,7,7,7,7,7,7,7,2,21,2,2,2,2,2,2,1,1,8,1,1,1,1,1,7,7,7,53,7,7,7,7,1,1,1,1,8,1,1,1,2,2,2,2,2,21,2,2,7,7,7,7,7,7,53,7,3,3,3,3,3,3,3,31'
a[4]='68,5,5,4,5,5,4,5,2,18,2,2,2,2,2,2,1,1,5,1,1,1,1,1,4,5,5,68,5,5,4,5,1,1,1,1,5,1,1,1,2,2,2,2,2,17,2,2,4,5,5,4,5,5,68,5,3,4,3,3,3,4,3,30'
a[5]='48,7,7,7,7,7,7,7,2,20,2,2,2,2,2,2,1,1,8,1,1,1,1,1,7,7,7,48,7,7,7,7,1,1,1,1,8,1,1,1,2,2,2,2,2,20,2,2,7,7,7,7,7,7,48,7,4,3,3,4,3,3,4,30'
a[6]='92,1,1,1,1,1,1,1,1,28,1,1,1,1,1,1,1,1,6,1,1,1,1,1,1,1,1,92,1,1,1,1,1,1,1,1,6,1,1,1,1,1,1,1,1,27,1,1,1,1,1,1,1,1,92,1,1,1,1,1,1,1,1,46'
a[7]='91,1,1,2,1,1,2,1,1,25,1,1,1,1,1,1,1,1,4,1,1,1,1,1,2,1,1,91,1,1,2,1,1,1,1,1,4,1,1,1,1,1,1,1,1,24,1,1,2,1,1,2,1,1,91,1,2,1,1,2,1,1,2,44'
a[8]='95,1,1,1,1,1,1,1,1,29,1,1,1,1,1,1,1,1,7,1,1,1,1,1,1,1,1,95,1,1,1,1,1,1,1,1,7,1,1,1,1,1,1,1,1,28,1,1,1,1,1,1,1,1,95,1,1,1,1,1,1,1,1,48'
a[9]='47,7,7,8,7,7,8,8,2,20,2,2,2,2,2,2,1,1,7,1,1,1,1,1,8,7,7,47,7,7,8,8,1,1,1,1,7,1,1,1,2,2,2,2,2,19,2,2,8,7,7,8,7,7,47,8,4,4,4,4,4,4,4,29'
a[10]='93,1,1,1,1,1,1,1,1,28,1,1,1,1,1,1,1,1,6,1,1,1,1,1,1,1,1,93,1,1,1,1,1,1,1,1,6,1,1,1,1,1,1,1,1,27,1,1,1,1,1,1,1,1,93,1,1,1,1,1,1,1,1,47'
a[11]='91,1,1,1,1,1,1,1,1,27,1,1,1,1,1,1,1,1,6,1,1,1,1,1,1,1,1,91,1,1,1,1,1,1,1,1,6,1,1,1,1,1,1,1,1,26,1,1,1,1,1,1,1,1,91,1,1,1,1,1,1,1,1,45'
a[12]='91,1,1,1,1,1,1,1,1,27,1,1,1,1,1,1,1,1,6,1,1,1,1,1,1,1,1,91,1,1,1,1,1,1,1,1,6,1,1,1,1,1,1,1,1,26,1,1,1,1,1,1,1,1,91,1,1,1,1,1,1,1,1,45'
a[13]='37,9,9,9,9,9,9,9,2,18,2,2,2,2,2,2,1,1,7,1,1,1,1,1,9,9,9,37,9,9,9,9,1,1,1,1,7,1,1,1,2,2,2,2,2,17,2,2,9,9,9,9,9,9,37,9,4,4,4,4,4,4,4,25'
a[14]='97,0,0,0,0,0,0,0,1,30,1,1,1,1,1,1,1,1,7,1,1,1,1,1,0,0,0,97,0,0,0,0,1,1,1,1,7,1,1,1,1,1,1,1,1,29,1,1,0,0,0,0,0,0,97,0,1,1,1,1,1,1,1,49'
a[15]='95,1,1,1,1,1,1,1,1,29,1,1,1,1,1,1,1,1,7,1,1,1,1,1,1,1,1,95,1,1,1,1,1,1,1,1,7,1,1,1,1,1,1,1,1,28,1,1,1,1,1,1,1,1,95,1,1,1,1,1,1,1,1,48'
a[16]='99,0,0,0,0,0,0,0,0,32,0,0,0,0,0,0,1,1,9,1,1,1,1,1,0,0,0,99,0,0,0,0,1,1,1,1,9,1,1,1,0,0,0,0,0,31,0,0,0,0,0,0,0,0,99,0,0,0,0,0,0,0,0,52'

c=1
for d in distances:
    for m in methods:
        temppp=a[c].split(',')
        for i,x in enumerate(temppp):
            temppp[i]=int(x)
        temppp=np.array(temppp).reshape(8,8)
        ConfMtr[m][d]=temppp
        c+=1

whoe_accuracyofclsuterings=[[0.4308,0.5676,0.5415,0.6298],[0.5192,0.8742,0.8519,0.8783],[0.4967,0.8753,0.8724,0.8724],[0.4167,0.9219,0.8783,0.9685]]

whoe_accuracyofclsuterings=np.array(whoe_accuracyofclsuterings)
whoe_accuracyofclsuterings=whoe_accuracyofclsuterings.transpose()


#%%
# =============================================================================
# =============================================================================
# # obtain the threshold for creating new cluster

### maximum distance between representatives
# =============================================================================
# =============================================================================
rep_dist=np.zeros((8,8))
for cl1 in representatives:
    print(cl1)
    for cl2 in representatives:
        rep_dist[cl1,cl2]=ccf(representatives[cl1],representatives[cl2],dds)
   #%%
# =============================================================================
# =============================================================================
# # Save the representative shapes from July 03 
# =============================================================================
# =============================================================================
representative_data={}
scale=6
shift=0   
SampleNum=40
for rep in representatives:
    if rep <8:
        anomm=representatives[rep]
        representative_data[rep]=dds[:,anomm*int(SampleNum/2)-40*scale+shift:(anomm*int(SampleNum/2)+40*scale+shift)]
    if rep==8:
        start,SampleNum,N=(0,40,500000)
        filename='data/Armin_Data/July_04/pkl/rawdata4.pkl'
        k=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG','TA', 'TB', 'TC']
        dds=load_standardized_data_with_features(filename,k)
        anomm=representatives[rep]
        representative_data[rep]=dds[:,anomm*int(SampleNum/2)-40*scale+shift:(anomm*int(SampleNum/2)+40*scale+shift)]

#detail about representatives
det_rep={3:[i for i in range(8)],4:[8],5:[],6:[],7:[],8:[],9:[]}

#%%
# =============================================================================
# =============================================================================
# # check the one day events (july 04) and make new clusters if it needed
# =============================================================================
# =============================================================================
#download the data for the considered day
total_event_cluster_data={}


ClusterNumber=len(representatives)
total_cluster_events={}
for i in representatives:
    total_cluster_events[i]=[]
for day in [4,5,6,7,8,9]:
    print(day)
    total_event_cluster_data[day]={}
    filename='data/Armin_Data/July_0'+str(day)+'/pkl/rawdata'+str(day)+'.pkl'
    k=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG','TA', 'TB', 'TC']
    data_04=load_standardized_data_with_features(filename,k)

    #detected events in this day
    event_folder_04='figures/all_events/July_0'+str(day)+'/GAN'
    events_04=os.listdir(event_folder_04)
    temp_ev_04=[]
    for i in events_04:
        temp_ev_04.append(i.split('.')[0])
    events_04=temp_ev_04

# =============================================================================
# =============================================================================
# # check each event in july 04 to representetives and if it's below the treshold make new cluster
# =============================================================================
# =============================================================================


    select_1224=data_04
    trh=0.14
    for count,event in enumerate(events_04):
        if count% 100==0:
            print('iter num: %i', count)
        event=int(event)
        #check the dist from representatives
        max_similarity=-1
        ClusterNumber=len(representatives)
        for candid in representative_data:
            sim=ccfWithRepresentatives(event,representative_data[candid],data_04)
            if sim>max_similarity:
                max_similarity=sim
                best_candidate=candid
        
        if max_similarity>trh:
            total_cluster_events[best_candidate].append(event)
            total_event_cluster_data[day][event]=best_candidate
#            print('event is: ',event,'nearest candidate: ',best_candidate,'similarity: ',max_similarity)
        else:      
            scale=6
            shift=0   
            SampleNum=40
            print('new cluter')
            print('new cluster is: ',event,'nearest candidate was: ',best_candidate,'similarity was: ',max_similarity)
            det_rep[day]=[ClusterNumber]
            representatives[ClusterNumber]=event
            representative_data[ClusterNumber]=data_04[:,event*int(SampleNum/2)-40*scale+shift:(event*int(SampleNum/2)+40*scale+shift)]
            total_cluster_events[ClusterNumber]=[event]
            total_event_cluster_data[day][event]=ClusterNumber
#            show([event],select_1224)
        
            
        
#%%
##cap bank cluster
#cap_jul4_ev=[471359,471360,471361,48493,48494,48495]
#ccfJul4CapBank=np.zeros((6,6))
#for x1,i in enumerate(cap_jul4_ev):
#    for x2,j in enumerate(cap_jul4_ev):
#        ccfJul4CapBank[x1,x2]=ccf(i,j,data_04)
#        
#
#index=np.argmax(sum(ccfJul4CapBank))
#candid=cap_jul4_ev[index]
#
#

#%%
   
   
   
   
    
# =============================================================================
# =============================================================================
# # Show each event we want from V, I and theta data
# =============================================================================
# =============================================================================
#select_1224=data_04
def showw(events,select_1224):
    SampleNum=40
    for anom in events:
            print(anom)
            anom=int(anom)
#            anom=events[anom]
#            print(anom)
            
            plt.subplot(221)
            for i in [0,1,2]:
                plt.plot(select_1224[i][anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)])
#            plt.legend('A' 'B' 'C')
            plt.title('V')
                
            plt.subplot(222)
            for i in [3,4,5]:
                plt.plot(select_1224[i][anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)])
#            plt.legend('A' 'B' 'C')
            plt.title('I')  
            
            plt.subplot(223)
            for i in [6,7,8]:
                plt.plot(select_1224[i][anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)])
#            plt.legend('A' 'B' 'C') 
#            figname=dst+"/"+str(anom)
#            plt.savefig(figname)
            plt.title('T')    
             
            plt.show()

#%%
# =============================================================================
# =============================================================================
# # save events figure in the same name folder but with V,I,T         
# =============================================================================
# =============================================================================
fn='clusters/cls/'
fnfolders=os.listdir(fn)
for f in fnfolders:
    clfolders=os.listdir(fn+f)
#    print(f)
    if f=='000000010' or f=='000000011' or f=='000000111':
        print(f)
        for cl in clfolders:
            showevents=[]
            imagelist=os.listdir(fn+f+'/'+cl)
            for ev in imagelist:
                showevents.append(int(ev.split('.')[0]))
            destination='clusters/vit/'+f+'/'+cl
            show(showevents,dd10,destination)
            print(destination)
            
#%%
# =============================================================================
# =============================================================================
# # mistakes
# =============================================================================
# =============================================================================
mistakes_folder='clusters/vit/mistakes/'
show([347468],dd13,mistakes_folder)
#%%    
# =============================================================================
# =============================================================================
# # Show each event we want from V, I and theta data
# =============================================================================
# =============================================================================

def show_representatives(rep):
#    SampleNum=40
    for anom in rep:
            print(len(total_cluster_events[anom]))
            print(anom)
#            anom=events[anom]
#            print(anom)
            
            plt.subplot(221)
            for i in [0,1,2]:
                plt.plot(rep[anom][i])
#            plt.legend('A' 'B' 'C')
            plt.title('V')
                
            plt.subplot(222)
            for i in [3,4,5]:
                plt.plot(rep[anom][i])
#            plt.legend('A' 'B' 'C')
            plt.title('I')  
            
            plt.subplot(223)
            for i in [6,7,8]:
                plt.plot(rep[anom][i])
#            plt.legend('A' 'B' 'C') 
            plt.title('T')    
             
            plt.show()
            #%%
anomalies

#%%%
            
        
            
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # # #             groupby feature detected event
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
#all events for July 03
whole_anoms=[]
for f in anomalies:
    whole_anoms.extend(anomalies[f])
whole_anoms=np.unique(whole_anoms)

#make embeded 9 features 0 and 1 for each event

July03_anomalies_detail={}
July03_anomalies_detail['event_time_chunk_number']=whole_anoms
July03_anomalies_detail['embeded_detection_features']=np.zeros((9,len(whole_anoms)))#9 is the independent number of features that we have
for idx,ev in enumerate(whole_anoms):
    if idx% 100==0:
        print(idx)
    for fnum,f in enumerate(anomalies):
        if np.isin(ev,anomalies[f]):
            July03_anomalies_detail['embeded_detection_features'][fnum,idx]=1
        else:
            July03_anomalies_detail['embeded_detection_features'][fnum,idx]=0
#%%
#now we seperate all events based on first step clustering
Stage_one_event_clusters={}
for i in range(len(July03_anomalies_detail['event_time_chunk_number'])):
    i=int(i)
    embdftr=July03_anomalies_detail['embeded_detection_features'][:,i]
    string=''
    for j in embdftr:
        string=string+str(int(j))
    if string in Stage_one_event_clusters:
        Stage_one_event_clusters[string].append(July03_anomalies_detail['event_time_chunk_number'][i])
    else:
        Stage_one_event_clusters[string]=[July03_anomalies_detail['event_time_chunk_number'][i]]
        
#%%
#inside each of these stge one clusters we should cluster them based on their simiarity
monitored_first_stage_clusters={}
monitored_first_stage_clusters['111111111']=Stage_one_event_clusters['111111111']#all features 3ph
monitored_first_stage_clusters['111000000']=Stage_one_event_clusters['111000000']#just V 3ph
monitored_first_stage_clusters['000111111']=Stage_one_event_clusters['000111111']# I and cos(theta) 3ph
# =============================================================================
monitored_first_stage_clusters['noise']=Stage_one_event_clusters['000000110']#noise cluster which is separated in the stage one
# =============================================================================
#monitored_first_stage_clusters['OnePhase']=Stage_one_event_clusters['111111111']
#%%
Stage_one_copy=Stage_one_event_clusters.copy()
for i in Stage_one_event_clusters:
    if len(Stage_one_copy[i])<15:
        
        del Stage_one_copy[i]
#%%
# =============================================================================
# =============================================================================
# # event clsuters in different days
# =============================================================================
# =============================================================================
cluster_per_day={}

for day in total_event_cluster_data.keys():
    cluster_per_day[day]={}
    selected_day_data=total_event_cluster_data[day]
    
    for ev in selected_day_data:
        cl=selected_day_data[ev]
        cl_in_day=list(cluster_per_day[day].keys())
        if cl in cl_in_day:
            cluster_per_day[day][cl].append(ev)
        else:
            cluster_per_day[day][cl]=[ev]
        
#%%
# =============================================================================
# =============================================================================
# # number of cluster events in different days
# =============================================================================
# =============================================================================
cl_def={0:'back to back',1:'current step down', 2:'signature',3:'med',4:'noise',5:'1 or 2 phases',6:'inrush',7:'med 2',8:'cap bank',9:'hifreq',10:'hifreq',11:'hifreq'}
for day in cluster_per_day:
    print('In July ',day,': ')
    for cl in cluster_per_day[day]:
        print('number of events in cluster ',cl_def[cl],' is ', len(cluster_per_day[day][cl]))
        
    
    
    