# =============================================================================
# =============================================================================
# =============================================================================
# # # save heatmap to show correlation between features
# =============================================================================
# =============================================================================
# =============================================================================
filename='data/Armin_Data/July_14/pkl/rawdata14.pkl'
k=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG','TA', 'TB', 'TC']
#dds14=load_standardized_data_with_features(filename,k)
dd14=load_data_with_features(filename,k)
start,SampleNum,N=(0,40,500000)
#%%
import seaborn as sn
id=[r'$\mid V_A \mid$',r'$\mid V_B \mid$',r'$\mid V_C \mid$',r'$\mid I_A \mid$',r'$\mid I_B \mid$',r'$\mid I_C \mid$',r'$cos(\theta_A)$',r'$cos(\theta_B)$',r'$cos(\theta_C)$']
corr=pd.DataFrame(np.corrcoef(dd3),index=id,columns=id)

sn.set(rc={'text.usetex': True})

#f, ax = plt.subplots(figsize=(16, 5))
#ax.set_ylabel('abc', rotation=0, fontsize=20, labelpad=20)
sn.set(font_scale=0.7)
#sn.plt.set_fontsize('18')
svm = sn.heatmap(corr,   
    cbar_kws={'fraction' : 0.1},
    linewidth=0.5, annot_kws={"size": 20})
svm.tick_params(labelsize=9)
#svm.set_xlabel(fontweight='bold')
svm.set_xticklabels(svm.get_xticklabels(), rotation=0,fontweight='bold',weight='bold')
svm.set_yticklabels(svm.get_yticklabels(), rotation=0, horizontalalignment='right',fontweight='bold',weight='bold')
plt.ylabel(r'$Features \ time \ series \ for \ a \ day$',fontweight='bold',fontsize=10)
plt.xlabel(r'$Features \ time \ series \ for \ a \ day$',fontweight='bold',fontsize=10)
#svm.ylabel('hi')

#plt.show() 

#ax.ylabel('hi')
#
figure = svm.get_figure()    
figure.savefig('journal/figures/heatmap.png',dpi=800)

#%%
# =============================================================================
# =============================================================================
# =============================================================================
# # # each case figure for cluster representative
# =============================================================================
# =============================================================================
# =============================================================================

dst='journal/figures'
def show_event(events,select_1224,dst):
    SampleNum=40
    for anom in events:
            print(anom)
            anom=int(anom)
#            anom=events[anom]
#            print(anom)
            space1=240
            space=240
            
            plt.subplot(311)
            for i in [0,1,2]:
                plt.plot(select_1224[i][anom*int(SampleNum/2)-space1:(anom*int(SampleNum/2)+space)])
            plt.legend('A' 'B' 'C')
            plt.ylabel(r'$|V|$',fontweight='bold',fontsize=10)

#            plt.title('V')
                
            plt.subplot(312)
            for i in [3,4,5]:
                from matplotlib.ticker import StrMethodFormatter
#                plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # No decimal places
                plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) # 2 decimal places
                plt.plot(select_1224[i][anom*int(SampleNum/2)-space1:(anom*int(SampleNum/2)+space)])
#            plt.legend('A' 'B' 'C')
            plt.ylabel(r'$|I| $',fontweight='bold',fontsize=10)
            
            plt.subplot(313)
            for i in [6,7,8]:
                plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
#                plt.plot(select_1224[i][anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)])
#            plt.legend('A' 'B' 'C')
                plt.plot(select_1224[i][anom*int(SampleNum/2)-space1:(anom*int(SampleNum/2)+space)])
#            plt.legend('A' 'B' 'C') 
            plt.ylabel(r'$cos(\theta)$',fontweight='bold',fontsize=10)  
            figname=dst+"/"+str(anom)
            plt.savefig(figname,dpi=800)
            
             
            plt.show()
            close()
            
    #%%
# =============================================================================
# =============================================================================
# =============================================================================
# # # extracting all events related to the inrush current for july 3
# =============================================================================
# =============================================================================
# =============================================================================


####inrush events
inrush=[]
for i in total_event_cluster_data[4]:
    if total_event_cluster_data[4][i]==6:
        inrush.append(i)
    #%%
    
    ###extract the magnitude and delta for each event
inrush_analysis={}

for i in inrush:
    
    anom=i
    wdata=dd4[:,anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)]
    
    tempwdata=wdata[:,200:400]
    curr=tempwdata[3,:]
    pf=tempwdata[6,:]
    m = max(curr)
    index=[i for i, j in enumerate(curr) if j == m][0]
    
    imax=m
    ibefore=curr[index-10]
    iafter=curr[index+10]
    
    
    
    m = min(pf)
    index=[i for i, j in enumerate(pf) if j == m][0]
    
    pfbefore=pf[index-10]
    pfafter=pf[index+10]
    
    
    
    inrush_analysis[i]=[imax-ibefore,iafter-ibefore,pfafter-pfbefore]
#%%

v=pd.DataFrame(inrush_analysis) 
#%%
#plt.scatter(v.iloc[0],v.iloc[1])
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x =v.iloc[0]
y =v.iloc[1]
z =v.iloc[2]



ax.scatter(x, y, z, c='r', marker='o')
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax.set_xlabel(r'$\Delta(I_{inrush})$',fontweight='bold',fontsize=10)
ax.set_ylabel(r'$\Delta(I_{steady \ state})$',fontweight='bold',fontsize=10)
ax.set_zlabel(r'$\Delta(cos(\theta)_{steady \ state})$',fontweight='bold',fontsize=10)

figname=dst+"/"+str('inrushscatter3d')
plt.savefig(figname,dpi=800)
            
plt.show()
 #%%

# =============================================================================
# =============================================================================
# # 3d inrush
# =============================================================================
# =============================================================================
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np

x =np.array(list(inrvalue.iloc[0]))
y =np.array(list(inrvalue.iloc[1]))
z =np.array(list(inrvalue.iloc[2]))
c=np.array(list(inrvalue.iloc[3]))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i,j in enumerate(markers):
    idata=inrvalue.loc[:,inrvalue.iloc[4]==j]
    x =np.array(list(idata.iloc[0]))
    y =np.array(list(idata.iloc[1]))
    z =np.array(list(idata.iloc[2]))
    c=np.array(list(idata.iloc[3]))

    
    ax.scatter(x,y,z,c=c)

#
#ax.scatter(x,y,z,c=c)
#
#x =v.iloc[0]
#y =v.iloc[1]
#z =v.iloc[2]
#
#plt.scatter(inrvalue.iloc[0],inrvalue.iloc[1],inrvalue.iloc[2],c=inrvalue.iloc[3])
#
##ax.scatter(x, y, z, c='r', marker='o')
#plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
#plt.xlabel(r'$\Delta(I_{inrush})$',fontweight='bold',fontsize=15)
#plt.ylabel(r'$\Delta(I_{steady \ state})$',fontweight='bold',fontsize=15)
#plt.ylabel(r'$\Delta(pf_{steady \ state})$',fontweight='bold',fontsize=15)
##ax.set_zlabel(r'$\Delta(cos(\theta)_{steady \ state})$',fontweight='bold',fontsize=10)
#
#figname=dst+"/"+str('inrushscatter2d')
#plt.savefig(figname,dpi=800)
            
plt.show()
#%%
# =============================================================================
# =============================================================================
# # inr event statistical figures
# =============================================================================
# =============================================================================

####medium events
inr_analysis={}
count=0
inr={}
colors=['r','b','c','k','g','y']
markers=['.','^','s','*','+','d']
d=0
for day in total_event_cluster_data:
    inr[day]=[]
    for i in total_event_cluster_data[day]:
        if total_event_cluster_data[day][i]==6:
            inr[day].append(i)
    filename='data/Armin_Data/July_0'+str(day)+'/pkl/rawdata'+str(day)+'.pkl'
    k=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG','TA', 'TB', 'TC']
    #dds4=load_standardized_data_with_features(filename,k)
    dayta=load_data_with_features(filename,k)

    
    ###extract the magnitude and delta for each event


    for i in inr[day]:
        
            
        anom=i
        wdata=dayta[:,anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)]
        
        tempwdata=wdata[:,200:400]
        curr=tempwdata[3,:]
        pf=tempwdata[6,:]
        m = max(curr)
        index=[i for i, j in enumerate(curr) if j == m][0]
        
        if index<170 and index>30: 
            imax=m
            ibefore=curr[index-30]
            iafter=curr[index+30]
        
        
        
        m = min(pf)
        index=[i for i, j in enumerate(pf) if j == m][0]
    
        if index<170 and index>30: 
            pfbefore=pf[index-30]
            pfafter=pf[index+30]
            
            color=colors[d]
            marker=markers[d]
            inr_analysis[count]=[imax-ibefore,iafter-ibefore,pfafter-pfbefore,color,marker,anom]
            count+=1
    d+=1
    #%%
inrvalue=pd.DataFrame(inr_analysis)
    
    
#%%
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.scatter(inrvalue.iloc[0],inrvalue.iloc[1]+1,c=inrvalue.iloc[3])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel(r'$\Delta(I_{inrush})$',fontweight='bold',fontsize=30)
plt.ylabel(r'$\Delta(I_{steady \ state})$',fontweight='bold',fontsize=30)
#figname=dst+"/"+str('inrushscatter2dwcolor7days')            
plt.show()
#plt.savefig(figname,dpi=800)
    
#%%
for i,j in enumerate(markers):
    idata=inrvalue.loc[:,inrvalue.iloc[4]==j]
    plt.scatter(idata.iloc[0],idata.iloc[1]+1,c=idata.iloc[3],s=20)
    
    
#%%
# =============================================================================
# =============================================================================
# # medium event statistical figures
# =============================================================================
# =============================================================================

####medium events
med_analysis={}
count=0
med={}

colors=['r','b','c','k','g','y']
markers=['.','^','s','*','+','d']
d=0
for day in total_event_cluster_data:
    med[day]=[]
    for i in total_event_cluster_data[day]:
        if total_event_cluster_data[day][i]==3:
            med[day].append(i)
    filename='data/Armin_Data/July_0'+str(day)+'/pkl/rawdata'+str(day)+'.pkl'
    k=['L1MAG','L2MAG', 'L3MAG','C1MAG','C2MAG', 'C3MAG','TA', 'TB', 'TC']
    #dds4=load_standardized_data_with_features(filename,k)
    dayta=load_data_with_features(filename,k)

    
    ###extract the magnitude and delta for each event


    for i in med[day]:
        
        anom=i
        wdata=dayta[:,anom*int(SampleNum/2)-240:(anom*int(SampleNum/2)+240)]
        
        tempwdata=wdata
        curr=tempwdata[3,:]
       
        mx = max(curr)
        mi = min(curr)
        
        mean=np.mean(curr)
        eps=0.2
        index=[]
        cr=0
        for i,j in enumerate(curr):
            if j>mean and cr==0:
                index.append(i)
                cr=1
            if cr==1 and j<=mean:
                cr=2
                index.append(i)
                
        if len(index)==2:
            
            if index[0]>10 and index[1]<470:
                before=curr[index[0]-10]
                after=curr[index[1]+10]
                durr=index[1]-index[0]
                color=colors[d]
                marker=markers[d]
                med_analysis[count]=[durr,after-before,color,marker,anom]
                count+=1
                
                
#            inr_analysis[count]=[imax-ibefore,iafter-ibefore,pfafter-pfbefore,color,marker,anom]
                count+=1
    d+=1
    #%%
medvalue=pd.DataFrame(med_analysis) 
    
    
#%%
plt.scatter(medvalue.iloc[0],medvalue.iloc[1])
    
#%%
for i,j in enumerate(markers):
    idata=medvalue.loc[:,medvalue.iloc[3]==j]
    plt.scatter(idata.iloc[0]+4,idata.iloc[1]+1,c=idata.iloc[2])
    
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel(r'$Duration (timeslots)$',fontweight='bold',fontsize=30)
plt.ylabel(r'$\Delta(I_{steady \ state})$',fontweight='bold',fontsize=30)
    
    
    