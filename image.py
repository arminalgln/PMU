window_data=j13[0][323438*20-4000:323438*20+4600]
#%%
window_median=np.median(window_data)
#%%
def mad_find(window_size,window_data,eps):
#    window_size=100
    shift=int(window_size/2)
#    shift=0
    data_size=window_data.shape[0]
    moving_median=[]
    MAD=[]
    upperbound=[]
    lowerbound=[]
    
    shift_moving_median=[]
    shift_MAD=[]
    shiftedup=[]
    shiftedlow=[]
    gama=1.4826
#    eps=5
    for window in range(int(data_size/window_size)):
        start=window*window_size
        end=start+window_size
        temp_data=window_data[start:end]
        temp_median=np.median(temp_data)
        
        temp_MAD=gama*np.median(np.absolute(temp_data-temp_median))
        
        moving_median.append(temp_median)
        MAD.append(temp_MAD)
        for i in range(window_size):
            upperbound.append(temp_median+eps*temp_MAD)
            lowerbound.append(temp_median-eps*temp_MAD)
            
        if window <int(data_size/window_size):
            start=window*window_size+shift
            end=start+window_size
            temp_data=window_data[start:end]
            temp_median=np.median(temp_data)
            
            temp_MAD=gama*np.median(np.absolute(temp_data-temp_median))
            
            shift_moving_median.append(temp_median)
            shift_MAD.append(temp_MAD)
            for i in range(window_size):
                shiftedup.append(temp_median+eps*temp_MAD)
                shiftedlow.append(temp_median-eps*temp_MAD)
            
    return lowerbound,upperbound,shiftedlow,shiftedup
#%%
ddtt=dd[7][2076500:207500]
    #%%
a=0
b=8000
plt.plot(window_data[a:b])
#for i in [120, 360, 600, 840, 1080]:
#    low,up,sl,su=mad_find(i,window_data,4.2)
#
#    plt.plot(low[a:b],color='r')
#    
#    plt.plot(up[a:b],color='r')
    
#    sl=np.array(sl)
#    sl=np.roll(sl,int(i/2))
#    su=np.array(su)
#    su=np.roll(su,int(i/2))
#    
#    plt.plot(sl,color='r')
#    
#    plt.plot(su,color='r')
#plt.title('Q (kVAR)',fontsize= 30)
plt.legend(['Voltage','Upper and lower bound'],fontsize=30)
plt.xlabel('Timeslots',fontsize= 30)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim([7125,7195)
#        plt.figtext(.5,.9,'Temperature', fontsize=100, ha='center')
#plt.xlabel('MPM',fontsize= 30)
plt.ylabel('Voltage (v)',fontsize= 30)
plt.show()
#%%






    #%%
plt.plot(window_data)
#for i in [120, 360, 600, 840, 1080]:
#    low,up,sl,su=mad_find(i,window_data,4.2)
#
#    plt.plot(low,color='r')
#    
#    plt.plot(up,color='r')
    
#    sl=np.array(sl)
#    sl=np.roll(sl,int(i/2))
#    su=np.array(su)
#    su=np.roll(su,int(i/2))
#    
#    plt.plot(sl,color='r')
#    
#    plt.plot(su,color='r')
#plt.title('Q (kVAR)',fontsize= 30)
plt.legend(['Voltage','Upper and lower bound'],fontsize=30)
plt.xlabel('Timeslots',fontsize= 30)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#        plt.figtext(.5,.9,'Temperature', fontsize=100, ha='center')
#plt.xlabel('MPM',fontsize= 30)
plt.ylabel('Voltage (v)',fontsize= 30)
plt.show()

















