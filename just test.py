# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:50:30 2019

@author: hamed
"""

Index(['Unnamed: 0', 'L1Mag', 'L2Mag', 'L3Mag', 'L1Ang', 'L2Ang', 'L3Ang',
       'C1Mag', 'C2Mag', 'C3Mag', 'C1Ang', 'C2Ang', 'C3Ang', 'PA', 'PB', 'PC',
       'QA', 'QB', 'QC'],
      dtype='object')
#%%

dir='data/Armin_Data/July_03'

foldernames=os.listdir(dir)
selected_files=np.array([])
for f in foldernames:
    spl=f.split('_')
    if 'Bld' in spl:
        selected_files=np.append(selected_files,f)
selected_files
filenames1224=natsort.natsorted(selected_files)
filenames1224
def OneFileImport(filename,dir):
    dir_name=dir
    base_filename=filename
    path=os.path.join(dir_name, base_filename)
    imported_data=pd.read_csv(path)
    return imported_data
whole_data_hun=np.array([])
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
    
    if count==0:
        whole_data_hun=selected_data.values
    else:
        whole_data_hun=np.append(whole_data_hun,selected_data.values,axis=0)
#%%
anom=2250900
sel=whole_data_hun[anom-240:anom+240]
c=0
for i in range(3):
    vm=sel[:,i+c+1]
    va=sel[:,i+4]-sel[:,4]
#    p=P2R(vm,va)
    
#    plt.plot(p.real,p.imag)
    plt.plot(vm)
plt.show()
sel=whole_data[anom-240:anom+240]

for i in range(3):
    vm=sel[:,i+c+1]
    va=sel[:,i+4]-sel[:,4]
#    p=P2R(vm,va)
    
#    plt.plot(p.real,p.imag)
    plt.plot(vm)

#%%
for i in range(3):
    vm=sel[:,i+1]
    va=sel[:,i+4]-sel[:,4]
    p=P2R(vm,va)
        
    plt.plot(p.real,p.imag)
    
    plt.show()
#%%
def P2R(radii, angles):
    return radii * np.exp(1j*angles)
#%%
p=P2R(vm,va)

fig,ax = plt.subplots()

ax.scatter(p.real,p.imag)

#%%
dir='data/Armin_Data/'
foldernames=os.listdir(dir)
filenames=natsort.natsorted(foldernames)
for fl in ['July_08']:
    print(fl)
    dist=dir+fl+'/pkl/J'+str(8)+'.pkl'
    pkl_file = open(dist, 'rb')
    selected_data = pkl.load(pkl_file)
    pkl_file.close()
    plt.plot(selected_data['1224']['L1MAG'])
    plt.show()
    









