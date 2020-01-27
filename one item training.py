# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:06:42 2019

@author: hamed
"""
mean=0

while mean!=4:
    rnd={}
    for i in range(epochnum):
        rnd[i]=np.random.randint(low=0,high=N,size=batch_size)
    
    #    show(rnd[i])
    
    generator= create_generator()
    discriminator= create_discriminator()
    gan = create_gan(discriminator, generator)
    
    kk=['TA']
    for idx,key in enumerate(kk):
        X_train_temp=X_train[:,(idx+6)]
    
    #X_train.reshape(N,3*SampleNum)
        X_train_temp=X_train_temp.reshape(N,SampleNum,1)
        tic = time.clock()   
        training(generator,discriminator,gan,epochnum,batch_size)
        toc = time.clock()
        print(toc-tic)
    #    
    #    gan_name='gan_sep_onelearn_good_09_'+key+'.h5'
    #    gen_name='gen_sep_onelearn_good_09_'+key+'.h5'
    #    dis_name='dis_sep_onelearn_good_09_'+key+'.h5'
    #    print(dis_name)
    #    gan.save(gan_name)
    #    generator.save(gen_name)
    #    discriminator.save(dis_name)
    scores_temp={}
    probability_mean={}
    anomalies_temp={}
    #kk=['TA','TB','TC']
    for idx,key in enumerate(kk):
        print(key)
        X_train_temp=X_train[:,(idx+6)]
    
    #X_train.reshape(N,3*SampleNum)
        X_train_temp=X_train_temp.reshape(N,SampleNum,1)
    
    #    id=int(np.floor(idx/3))
    #    mode=k[id*3]
    #    dis_name='dis_sep_onelearn_'+mode+'.h5'
    #    
    #    discriminator=load_model(dis_name)
        
        
        rate=1000
        shift=N/rate
        scores_temp[key]=[]
        for i in range(rate-1):
            temp=discriminator.predict_on_batch(X_train_temp[int(i*shift):int((i+1)*shift)])
            scores_temp[key].append(temp)
            print(i)
        
        scores_temp[key]=np.array(scores_temp[key])
        scores_temp[key]=scores_temp[key].ravel()
        
        probability_mean[key]=np.mean(scores_temp[key])
        data=scores_temp[key]-probability_mean[key]
        
        mu, std = norm.fit(data)
        
        zp=3
        
        high=mu+zp*std
        low=mu-zp*std
        
        anomalies_temp[key]=np.union1d(np.where(data>=high)[0], np.where(data<=low)[0])
        print(anomalies_temp[key].shape)
        
        mean=np.mean(scores_temp['TA'])
        mean=np.floor(int(mean*10))