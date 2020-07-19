# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 07:03:33 2019

@author: Ivan Petej
"""

# Python code for the Neurocomuting paper submission 

# "Conformal calibrators", COPA, Spring 2020

# import required libraries
import numpy as np
#from sklearn.model_selection import KFold
 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as tfk
tfd = tfp.distributions

import math

from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

import random

#---------------------------CRPS/SCC FUNCTIONS-------------------------------

# define Continuous Ranked Probability Distribution (CRPS) calculation
def crps(c_i,y_test,typ='point'): 
    # CRPS calculation
    
    # For point predictions  
    if(typ=='point'):     
        [u,indices,counts]=np.unique((c_i),return_index=True,return_counts=True)
        counts=np.asarray(counts)
        q=np.empty((len(u),3))
        for i in range(len(u)):
            q[i,0]=u[i]
            q[i,1]=np.sum(counts[:i+1])/len(c_i)
        
            if(u[i]<y_test):
                q[i,2]=0
            else:
                q[i,2]=1
            
        crps_mat=q.copy()
    
        if(np.isin(y_test,crps_mat[:,0])==False):
            if(u[0]>y_test):        
                crps_mat=np.vstack((np.asarray([y_test,0,1]),crps_mat))
            elif(u[-1]<y_test):        
                crps_mat=np.vstack((crps_mat,np.asarray([y_test,1,1])))   
            else:
                ind=np.argmax(crps_mat[:,0]>y_test)
                crps_mat=np.vstack((crps_mat[:ind,:],np.asarray([y_test,crps_mat[ind-1,1],1]),crps_mat[ind:,:]))
    
        crps=np.sum((crps_mat[0:-1,1]-crps_mat[0:-1,2])**2 * np.diff(crps_mat[:,0]))
        if(q[0,2]==1):
            pY=q[0,1]
        elif(q[-1,2]==0):
            pY=q[-1,1]
        else:
            pY = min(q[q[:,2]==1,1])
            
            
    # For probabilistic predictions   
    else:     
        [u,indices,counts]=np.unique((c_i[:,1]),return_index=True,return_counts=True)
        counts=np.asarray(counts)
        q=np.empty((len(u),3))
        for i in range(len(u)):
            q[i,0]=u[i]
            q[i,1]=c_i[indices[i],0]
        
            if(u[i]<y_test):
                q[i,2]=0
            else:
                q[i,2]=1
            
        crps_mat=q.copy()
    
        if(np.isin(y_test,crps_mat[:,0])==False):
            if(u[0]>y_test):        
                crps_mat=np.vstack((np.asarray([y_test,0,1]),crps_mat))
            elif(u[-1]<y_test):        
                crps_mat=np.vstack((crps_mat,np.asarray([y_test,1,1])))   
            else:
                ind=np.argmax(crps_mat[:,0]>y_test)
                crps_mat=np.vstack((crps_mat[:ind,:],np.asarray([y_test,crps_mat[ind-1,1],1]),crps_mat[ind:,:]))
    
        crps=np.sum((crps_mat[0:-1,1]-crps_mat[0:-1,2])**2 * np.diff(crps_mat[:,0]))
        if(q[0,2]==1):
            pY=q[0,1]
        elif(q[-1,2]==0):
            pY=q[-1,1]
        else:
            pY = min(q[q[:,2]==1,1])
    
    return(q,pY,crps)
    
# define function for crps summary calculation
    
def derive_crps(n_cal,n_test,y_cal,y_test,y_hat_cal,y_hat_test,y_cal_dist,y_test_dist):
    
    # Define storage (0 point, 1 pd, 2 cps, 3 cc)
        crps_store=np.zeros((n_test,4))
        F_i_store=np.zeros((n_test,4))          
        alpha_i=np.zeros(n_cal)
        
    # Calculate y grid from calibration distribution
    
    # Calculate alphas on the calibration set  
        print('Calculating alphas on the calibration set...')
        for i in range(n_cal):
            [u,indices,counts]=np.unique(y_cal_dist[i,:],return_index=True,return_counts=True)
            counts=np.asarray(counts)
            q=np.empty((len(u),2))
            for j in range(len(u)):
                q[j,0]=u[j]
                q[j,1]=np.sum(counts[:j+1])/len(y_cal_dist[i,:])
            if(y_cal[i]>u[-1]):
                alpha_i[i]=1
            elif(y_cal[i]<u[0]):
                alpha_i[i]=0
            else:
                if len(q[:,0])>1:
                    f = interp1d(q[:,0], q[:,1])
                    alpha_i[i] = f(y_cal[i])
                else:
                    alpha_i[i] = q[0,1]
                    
        print('done.')
           
        # Generate SCPS on the test set
        print('Generate SCPS on the test set...')
        c_i=np.zeros((n_test,n_cal))
        
        for i in range(n_test):
            
            print('test example ' + str(i) + '/' + str(n_test), end='')
            print('\n', end='')
            
            [u,indices,counts]=np.unique(y_test_dist[i,:],return_index=True,return_counts=True)
            counts=np.asarray(counts)
            q=np.empty((len(u),2))
            c_a=np.empty(len(u))
            for j in range(len(u)):
                q[j,0]=u[j]
                q[j,1]=np.sum(counts[:j+1])/len(y_test_dist[i,:])
                c_a[j]=1/(n_cal+1)*len(np.transpose(np.where(alpha_i[:] < q[j,1])))+np.random.uniform(0,1)/(n_cal+1)*(1+len(np.transpose(np.where(alpha_i[:] == q[j,1]))))
        
            if(y_test[i]>u[-1]):
                F_i_store[i,0]=1
                F_i_store[i,1]=1
            elif(y_test[i]<u[0]):
                F_i_store[i,0]=0
                F_i_store[i,1]=0
            else:
                if len(q[:,0])>1:
                    f = interp1d(q[:,0], q[:,1])
                    F_i_store[i,2] = f(y_test[i])
                    f = interp1d(q[:,0], c_a)
                    F_i_store[i,3] = f(y_test[i])
                else:
                     F_i_store[i,2] = q[0,1]
                     F_i_store[i,3] = q[0,1]
                
        
            # Calculate CRPS for point, underlying algorithm and SCPD predictions 
            crps_store[i,0]=np.abs(y_hat_test[i]-y_test[i])
            
            c_i[i,:] = y_hat_test[i]+(y_cal.flatten() - y_hat_cal.flatten()) # compute alphas C on the calibrations set 
            (q_cps, pY_cps, crps_cps)=crps(c_i[i,:],y_test[i],'point')
            crps_store[i,1]=crps_cps
            F_i_store[i,1] = pY_cps
            [q_pd,_,crps_pd]=crps(np.transpose(np.vstack((np.transpose(q[:,1]),np.transpose(q[:,0])))),y_test[i],'dist')
            crps_store[i,2]=crps_pd
            [q_scc,_,crps_scc]=crps(np.transpose(np.vstack((np.transpose(c_a),np.transpose(q[:,0])))),y_test[i],'dist')
            crps_store[i,3]=crps_scc
            
        return(crps_store,F_i_store)


def calc_dist(model,model_best,x_train,y_train,x_cal,y_cal,x_test,y_test):
    
    if(model == 'RF'):
        # RF
        model_best.fit(x_train,y_train)   
        scratch=[tree.predict(x_cal) for tree in model_best.estimators_]  
        y_cal_dist=np.transpose(np.asarray(scratch))
        y_hat_cal=model_best.predict(x_cal)
            
        scratch=[tree.predict(x_test) for tree in model_best.estimators_]  
        y_test_dist=np.transpose(np.asarray(scratch))
        y_hat_test=model_best.predict(x_test)
            
    elif(model == 'GLP_MAT'or model == 'GLP_RBF'): # GLP
        # GLP    
        model_best.fit(x_train,y_train)   
        y_hat_cal, sigma_hat_cal = model_best.predict(x_cal, return_std=True)
        y_hat_test, sigma_hat_test = model_best.predict(x_test, return_std=True)
        y_cal_dist = np.array([np.random.normal(m, s, 1000) for m, s in zip(y_hat_cal, sigma_hat_cal)],)
        y_test_dist = np.array([np.random.normal(m, s, 1000) for m, s in zip(y_hat_test, sigma_hat_test)])
    else:
        #TF
        model_best.fit(x_train, y_train, epochs=100, verbose=False)    
        y_cal_tf = model_best(x_cal)
        y_test_tf=model_best(x_test)
    
        y_hat_cal = y_cal_tf.mean().numpy()
        sigma_hat_cal = np.abs(y_cal_tf.stddev().numpy())
        y_hat_test = y_test_tf.mean().numpy()
        sigma_hat_test = np.abs(y_test_tf.stddev().numpy())
        
        y_cal_dist = np.array([np.random.normal(m, s, 1000) for m, s in zip(y_hat_cal, sigma_hat_cal)],)
        y_test_dist = np.array([np.random.normal(m, s, 1000) for m, s in zip(y_hat_test, sigma_hat_test)])
 
    return (y_hat_cal,y_cal_dist,y_hat_test,y_test_dist)

     
# data import function (note datasets are csv files in the home directory, or python generic)   
#---------------------------------------DATA-----------------------------------------------

def getDta(dtaString):

    
    if (dtaString=='artificial_het'):
        
        z = 500
    
        n = 400
        m = 200
        
        n_cal = n - m
        n_test = z - n
        
        
        X = np.random.uniform(-1, 1, z)
        Y = 2 * X + np.random.normal(0, abs(X)/2, z)
        
        X=X.reshape(-1,1)
        
        # training, calibration and test set objects
        x_train = X[:m] # training set objects
        x_cal   = X[m:n] # calibration objects
        x_test  = X[n:] # test set objects  
                            
        y_train = Y[:m] # training set labels
        y_cal =   Y[m:n]  # validation set labels
        y_test =  Y[n:] # test set labels

        x_train_cal = X[:n] # objects from training-calibration set
        y_train_cal = Y[:n] # labels training-calibration set
        
    elif (dtaString=='artificial_het_cov_2'):
        
        z = 500
    
        n = 400
        m = 200
        
        n_cal = n - m
        n_test = z - n
        
        x_train_cal = np.random.uniform(-1, 0, n)
        x_test=np.random.uniform(0, 1, n_test)
        y_train_cal = 2 * x_train_cal + np.random.normal(0, abs(x_train_cal)/2, n)
        y_test = 2 * x_test + np.random.normal(0, abs(x_test/2), n_test)
        
        x_train_cal = x_train_cal.reshape(-1,1)
        x_test = x_test.reshape(-1,1)
        
        # training, calibration objects
        x_train   = x_train_cal[:m] # calibration objects
        x_cal   = x_train_cal[m:n] # calibration objects
                            
        y_train = y_train_cal[:m] # training set labels
        y_cal =   y_train_cal[m:n]  # validation set labels
        
        
        X = np.vstack((x_train_cal,x_test))
        Y = np.vstack((y_train_cal.reshape(-1,1),y_test.reshape(-1,1)))
        
    elif (dtaString=='artificial_norm'):
        
        z = 500
    
        n = 400
        m = 200
        n_cal = n - m
        n_test = z - n
        
        X = np.random.uniform(-1, 1, z)
        Y = 2 * X + np.random.normal(0, 0.5/2, z)
        
        X=X.reshape(-1,1)
        
        # training, calibration and test set objects
        x_train = X[:m] # training set objects
        x_cal   = X[m:n] # calibration objects
        x_test  = X[n:] # test set objects  
                            
        y_train = Y[:m] # training set labels
        y_cal =   Y[m:n]  # validation set labels
        y_test =  Y[n:] # test set labels

        x_train_cal = X[:n] # objects from training-calibration set
        y_train_cal = Y[:n] # labels training-calibration set
        
    elif (dtaString=='artificial_het_cov_1'):
        
        z = 500
    
        n = 400
        m = 200
        
        n_cal =  n - m
        n_test =  z - n
        
        x_train_cal = np.random.uniform(-1, 0, n)
        x_test=np.random.uniform(0, 1, n_test)
        y_train_cal = 2 * x_train_cal + np.random.normal(0, 0.5, n)
        y_test = 2 * x_test + np.random.normal(0, 2, n_test)
        
        x_train_cal = x_train_cal.reshape(-1,1)
        x_test = x_test.reshape(-1,1)
        
        # training, calibration objects
        x_train   = x_train_cal[:m] # calibration objects
        x_cal   = x_train_cal[m:n] # calibration objects
                            
        y_train = y_train_cal[:m] # training set labels
        y_cal =   y_train_cal[m:n]  # validation set labels
        
        X = np.vstack((x_train_cal,x_test))
        Y = np.vstack((y_train_cal.reshape(-1,1),y_test.reshape(-1,1)))


        
    elif(dtaString=='sl'):
        
        
        train = np.load('./data/train_shuffle.npy')
        test = np.load('./data/test_shuffle.npy')
        
        # randomly select 1,000 training examples and 100 test examples           
        ind = np.random.choice(len(train),1000,replace=False)
        train = train[ind]
              
        ind = np.random.choice(len(test),100,replace=False)
        test = test[ind]
        
     
        n = len(train)    # training set
        m = int(n/2)    #proper training set

        n_cal = n - m 
        n_test = len(test)
        
        z = n + n_test
        
        x_train_cal = train[:,:-1]
        y_train_cal = train[:,-1]
        
        x_test = test[:,:-1]
        y_test = test[:,-1]
        
  # training, calibration objects
        x_train   = x_train_cal[:m] # calibration objects
        x_cal   = x_train_cal[m:n] # calibration objects
                            
        y_train = y_train_cal[:m] # training set labels
        y_cal =   y_train_cal[m:n]  # validation set labels
        
        X = np.vstack((x_train_cal,x_test))
        Y = np.vstack((y_train_cal.reshape(-1,1),y_test.reshape(-1,1)))

    
    return(X,Y,n,m,n_cal,n_test,x_train,x_cal,x_test,y_train,y_cal,y_test,x_train_cal,y_train_cal)
    

# set seed

np.random.seed(1)

# define dataset
dset='artificial_norm' 

#set proper  and calibration ratios for the training set
cal_train_ratio = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

# define the total number of folds
fold_K = len(cal_train_ratio)

# specify number of simulations
L = 1

models=['RF','GLP_MAT', 'GLP_RBF', 'TF']

[X,Y,n,m,n_cal,n_test,x_train,x_cal,x_test,y_train,y_cal,y_test,x_train_cal,y_train_cal]=getDta(dset)
    

if dset != 'sl':
    
    fig = plt.figure(figsize=(11.69,8.27))

    ax1 = fig.add_subplot(111)
    ax1.scatter(X,Y)
    ax1.set_title(dset)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    fig.savefig('./figures/data_' + dset +'.pdf', bbox_inches='tight')

    plt.show()

#---------------------------------------MAIN LOOP------------------------------------------------

for idx, model in enumerate(models):
    
# get data
    
    [X,Y,n,m,n_cal,n_test,x_train,x_cal,x_test,y_train,y_cal,y_test,x_train_cal,y_train_cal]=getDta(dset)
    
    # define CRPS and probability storage variables
    
    # point predictions
    CRPS_point = np.empty((n_test*L,fold_K))
    CRPS_point[:] = np.NaN
    
    # conformal predictive system
    CRPS_cps = np.empty((n_test*L,fold_K))
    CRPS_cps[:] = np.NaN
    pY_cps=np.empty((n_test*L,fold_K))
    
    # underlying probability distribution of base algorithm
    CRPS_pd = np.empty((n_test*L,fold_K))
    CRPS_pd[:] = np.NaN
    pY_pd=np.empty((n_test*L,fold_K))
    
    # split conformal calibrators
    CRPS_scc = np.empty((n_test*L,fold_K))
    CRPS_scc[:] = np.NaN
    pY_scc=np.empty((n_test*L,fold_K))
    
    # cross conformal calibratros
    CRPS_ccc = np.empty((n_test*L,fold_K))
    CRPS_ccc[:] = np.NaN
    pY_ccc=np.empty((n_test*L,fold_K))
    
    
    # repeat over number of different siumlations
    for l in range(L):   
        # get data
        [X,Y,n,m,n_cal,n_test,x_train,x_cal,x_test,y_train,y_cal,y_test,x_train_cal,y_train_cal]=getDta(dset)
    
     
        if(model == 'RF'): # RF first
         # Random forests first
            print('--------------------RF-----------------------')
            # Setting up network
            print("Setting up Params...")
            
            n_estimators = [100,500,1000]
            max_features = ['auto', 'sqrt']
            max_depth = [10,50,100,200]
            min_samples_split = [2,5]
            min_samples_leaf = [1,2]
            bootstrap = [True, False]   
        
            # Create the random grid
            random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}  
            
            print("done.")
        
            # Define RF regression and train model
            print("Training model...")
                
            rf = RandomForestRegressor()
            rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=10, random_state=1, n_jobs = 1)
            rf_random.fit(x_train_cal, y_train_cal)
            model_best = RandomForestRegressor()
            model_best.set_params(**rf_random.best_params_)
        
        elif(model  == 'GLP_MAT'):          
             # GLP matern next
            print('--------------------GLP_MATERN-----------------------') 
             # Setting up network
            print("Setting up Params...") 
            kernel = ConstantKernel() * Matern(length_scale=x_train_cal.shape[1]) \
            + WhiteKernel()
            print("done.")
    
            # Define GLP regression and train model
            print("Training model...")
            gp = gaussian_process.GaussianProcessRegressor(kernel=kernel,alpha=0, n_restarts_optimizer = 10)
            gp.fit(x_train_cal, y_train_cal)
            model_best = gp  
            print("done.")
            
        elif(model == 'GLP_RBF'):           
            # RBF next
            print('--------------------GLP_RBF-----------------------') 
             # Setting up network
            print("Setting up Params...") 
            kernel = ConstantKernel() * RBF(length_scale=x_train_cal.shape[1]) \
            + WhiteKernel()
            print("done.")
    
            # Define GLP regression and train model
            print("Training model...")
            gp = gaussian_process.GaussianProcessRegressor(kernel=kernel,alpha=0, n_restarts_optimizer = 10)
            gp.fit(x_train_cal, y_train_cal)
            model_best = gp  
            print("done.")
        else:
            # Tensorflow
            print('--------------------TF_PROB-----------------------') 
            # Setting up network
            print("Setting up Params...") 
            n_features = x_train_cal.shape[1]
            tf_model = tfk.Sequential([
                    tf.keras.layers.Dense(n_features),
                    tf.keras.layers.Dense(math.ceil(n_features / 2.) * 2), 
                    tf.keras.layers.Dense(1 + 1), 
#                    tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :], scale=100))])
                    tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1],scale=np.average(y_train_cal) + tf.math.softplus(np.std(y_train_cal)/100 * t[..., 1:]))),])
            tf_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=tf.keras.losses.MeanSquaredError())
            print("done.")
        
            # Define GLP regression and train model
            print("Training model...")
            tf_model.fit(x_train_cal, y_train_cal, epochs=100, verbose=False)
            model_best = tf_model
            print("done.")
               
            
    #    for each train/calibration split and each fold
        
        for k in range(fold_K):
            
            
            print (model + str(l) + '_' + str(k))
            
    #        split data
            ind = np.arange(0,n)
            
            random.shuffle(ind)
            
            train_ind = ind[:int(n * (1-cal_train_ratio[k]))]
            cal_ind = ind[int(n * (1-cal_train_ratio[k])):]
            
            x_train = x_train_cal[train_ind]
            x_cal   = x_train_cal[cal_ind]
            y_train = y_train_cal[train_ind]
            y_cal   = y_train_cal[cal_ind]
            
            n_cal = len(y_cal)
            
            [y_hat_cal,y_cal_dist,y_hat_test,y_test_dist] = calc_dist(model,model_best,x_train,y_train,x_cal,y_cal,x_test,y_test)
            
            # calculate crps and F values
            crps_store, F_i_store = derive_crps(n_cal,n_test,y_cal,y_test,y_hat_cal,y_hat_test,y_cal_dist,y_test_dist)    
                      
            # store
            
            CRPS_point[n_test*l:n_test*(l+1),k] = crps_store[:,0]
            
            CRPS_cps[n_test*l:n_test*(l+1),k] = crps_store[:,1]
            pY_cps[n_test*l:n_test*(l+1),k] = F_i_store[:,1]
    
            # underlying probability distribution of base algorithm
            CRPS_pd[n_test*l:n_test*(l+1),k] = crps_store[:,2]
            pY_pd[n_test*l:n_test*(l+1),k] = F_i_store[:,2]
    
            # split calibrated probability distribution
            CRPS_scc[n_test*l:n_test*(l+1),k] = crps_store[:,3]
            pY_scc[n_test*l:n_test*(l+1),k] = F_i_store[:,3]
                

       
    
    # save data
    
    np.savetxt("./figures/CRPS_point_" + model + "_" + dset + ".csv", CRPS_point, delimiter=",")
    np.savetxt("./figures/CRPS_cps_" + model + "_" + dset + ".csv", CRPS_cps, delimiter=",")
    np.savetxt("./figures/CRPS_pd_" + model + "_" + dset + ".csv", CRPS_pd, delimiter=",")
    np.savetxt("./figures/CRPS_scc_" + model + "_" + dset + ".csv", CRPS_scc, delimiter=",")
    np.savetxt("./figures/CRPS_ccc_" + model + "_" + dset + ".csv", CRPS_ccc, delimiter=",")
    np.savetxt("./figures/pY_cps_" + model + "_" + dset + ".csv", pY_cps, delimiter=",")
    np.savetxt("./figures/pY_pd_" + model + "_" + dset + ".csv", pY_pd, delimiter=",")
    np.savetxt("./figures/pY_scc_" + model + "_" + dset + ".csv", pY_scc, delimiter=",")
    np.savetxt("./figures/pY_ccc_" + model + "_"  + dset + ".csv", pY_ccc, delimiter=",")  

 





