"""
Created on Sun Apr 12 15:32:04 2020

@author: ivanpetej
"""

import numpy as np
import matplotlib.pyplot as plt

dsets = ['artificial_norm','artificial_het','artificial_het_cov_1','artificial_het_cov_2']
names = ['Norm','Het','HetCov_1','HetCov_2']

fig = plt.figure(figsize=(11.69,10.27))

for idx,dset in enumerate(dsets):

    CRPS_scc_RF=np.genfromtxt('./figures/CRPS_scc_RF_' + dset + '.csv', delimiter=',')
    CRPS_cps_RF=np.genfromtxt('./figures/CRPS_cps_RF_' + dset + '.csv', delimiter=',')


    ax1 = fig.add_subplot(2,2,idx+1)
    ax1.scatter(CRPS_cps_RF.flatten(),CRPS_scc_RF.flatten(),s=4)
    ax1.plot(CRPS_cps_RF.flatten(),CRPS_cps_RF.flatten(),'k--')
    ax1.set_title(names[idx])
    ax1.set_ylim([0,0.5+int(np.max([CRPS_cps_RF.flatten(),CRPS_scc_RF.flatten()]))])
    ax1.set_xlim([0,0.5+int(np.max([CRPS_cps_RF.flatten(),CRPS_scc_RF.flatten()]))])

fig.savefig('./figures/error_dependence.pdf', bbox_inches='tight')

plt.show()


# ---- calibration plot for Stena Line dataset ---------------------

dset='artificial_het_cov_2'

if dset=='sl':
    
    # calibration plot
    
    pY_RF_pd=np.genfromtxt('./figures/pY_pd_RF_' + dset + '.csv', delimiter=',')
    pY_GLP_RBF_pd=np.genfromtxt('./figures/pY_pd_GLP_RBF_' + dset + '.csv', delimiter=',')
    pY_GLP_MAT_pd=np.genfromtxt('./figures/pY_pd_GLP_MAT_' + dset + '.csv', delimiter=',')
    pY_TF_pd=np.genfromtxt('./figures/pY_pd_TF_' + dset + '.csv', delimiter=',')
    pY_RF_scc=np.genfromtxt('./figures/pY_scc_RF_' + dset + '.csv', delimiter=',')
    pY_GLP_RBF_scc=np.genfromtxt('./figures/pY_scc_GLP_RBF_' + dset + '.csv', delimiter=',')
    pY_GLP_MAT_scc=np.genfromtxt('./figures/pY_scc_GLP_MAT_' + dset + '.csv', delimiter=',')
    pY_TF_scc=np.genfromtxt('./figures/pY_scc_TF_' + dset + '.csv', delimiter=',')
    
    # define calibration plot variables
    p_RF_pd=np.zeros(100)
    p_GLP_RBF_pd=np.zeros(100)
    p_GLP_MAT_pd=np.zeros(100)
    p_TF_pd=np.zeros(100)
    p_RF_scc=np.zeros(100)
    p_GLP_RBF_scc=np.zeros(100)
    p_GLP_MAT_scc=np.zeros(100)
    p_TF_scc=np.zeros(100)
    
    # calculate calibration variables
    for i in range(100):
        p_RF_pd[i]=1-np.sum(pY_RF_pd[:,1].reshape(-1,1) > i/100)/len(pY_RF_pd[:,1].reshape(-1,1))
        p_GLP_RBF_pd[i]=1-np.sum(pY_GLP_RBF_pd[:,1].reshape(-1,1) > i/100)/len(pY_GLP_RBF_pd[:,1].reshape(-1,1))
        p_GLP_MAT_pd[i]=1-np.sum(pY_GLP_MAT_pd[:,1].reshape(-1,1) > i/100)/len(pY_GLP_MAT_pd[:,1].reshape(-1,1))
        p_TF_pd[i]=1-np.sum(pY_TF_pd[:,1].reshape(-1,1) > i/100)/len(pY_TF_pd[:,1].reshape(-1,1))
        p_RF_scc[i]=1-np.sum(pY_RF_scc[:,1].reshape(-1,1) > i/100)/len(pY_RF_scc[:,1].reshape(-1,1))
        p_GLP_RBF_scc[i]=1-np.sum(pY_GLP_RBF_scc[:,1].reshape(-1,1) > i/100)/len(pY_GLP_RBF_scc[:,1].reshape(-1,1))
        p_GLP_MAT_scc[i]=1-np.sum(pY_GLP_MAT_scc[:,1].reshape(-1,1) > i/100)/len(pY_GLP_MAT_scc[:,1].reshape(-1,1))
        p_TF_scc[i]=1-np.sum(pY_TF_scc[:,1].reshape(-1,1) > i/100)/len(pY_TF_scc[:,1].reshape(-1,1))
        
        
    # plot calibration figures
    fig = plt.figure(figsize=(11.69,8.27))
    
    ax1 = fig.add_subplot(221)
    ax1.plot(np.arange(0, 1, 0.01),p_RF_pd,'r-',label='base')
    ax1.plot(np.arange(0, 1, 0.01),p_RF_scc,'b-',label='aSCPS')
    ax1.plot(np.arange(0, 1, 0.01),np.arange(0, 1, 0.01),'k--')
    ax1.legend()
    ax1.set_title('RF')
    
    ax2 = fig.add_subplot(222)
    ax2.plot(np.arange(0, 1, 0.01),p_GLP_MAT_pd,'r-',label='base')
    ax2.plot(np.arange(0, 1, 0.01),p_GLP_MAT_scc,'b-',label='aSCPS')
    ax2.plot(np.arange(0, 1, 0.01),np.arange(0, 1, 0.01),'k--')
    ax2.legend()
    ax2.set_title('GM')
    
    
    ax3 = fig.add_subplot(223)
    ax3.plot(np.arange(0, 1, 0.01),p_GLP_RBF_pd,'r-',label='base')
    ax3.plot(np.arange(0, 1, 0.01),p_GLP_RBF_scc,'b-',label='aSCPS')
    ax3.plot(np.arange(0, 1, 0.01),np.arange(0, 1, 0.01),'k--')
    ax3.legend()
    ax3.set_title('GRBF')
    
    
    ax4 = fig.add_subplot(224)
    ax4.plot(np.arange(0, 1, 0.01),p_TF_pd,'r-',label='base')
    ax4.plot(np.arange(0, 1, 0.01),p_TF_scc,'b-',label='aSCPS')
    ax4.plot(np.arange(0, 1, 0.01),np.arange(0, 1, 0.01),'k--')
    ax4.legend()
    ax4.set_title('TF')
    
    
    
    plt.show()
    
    fig.savefig('./figures/calibration_' + dset+'_base.pdf', bbox_inches='tight')


## CRPS error tables

    # import data
CRPS_cps_RF=np.genfromtxt('./figures/CRPS_cps_RF_' + dset + '.csv', delimiter=',')
CRPS_pd_RF=np.genfromtxt('./figures/CRPS_pd_RF_' + dset + '.csv', delimiter=',')
CRPS_cps_GLP_RBF=np.genfromtxt('./figures/CRPS_cps_GLP_RBF_' + dset + '.csv', delimiter=',')
CRPS_pd_GLP_RBF=np.genfromtxt('./figures/CRPS_pd_GLP_RBF_' + dset + '.csv', delimiter=',')
CRPS_cps_GLP_MAT=np.genfromtxt('./figures/CRPS_cps_GLP_MAT_' + dset + '.csv', delimiter=',')
CRPS_pd_GLP_MAT=np.genfromtxt('./figures/CRPS_pd_GLP_MAT_' + dset + '.csv', delimiter=',')
CRPS_cps_TF=np.genfromtxt('./figures/CRPS_cps_TF_' + dset + '.csv', delimiter=',')
CRPS_pd_TF=np.genfromtxt('./figures/CRPS_pd_TF_' + dset + '.csv', delimiter=',')

pY_RF_pd=np.genfromtxt('./figures/pY_pd_RF_' + dset + '.csv', delimiter=',')
pY_GLP_RBF_pd=np.genfromtxt('./figures/pY_pd_GLP_RBF_' + dset + '.csv', delimiter=',')
pY_GLP_MAT_pd=np.genfromtxt('./figures/pY_pd_GLP_MAT_' + dset + '.csv', delimiter=',')
pY_TF_pd=np.genfromtxt('./figures/pY_pd_TF_' + dset + '.csv', delimiter=',')
pY_RF_cps=np.genfromtxt('./figures/pY_cps_RF_' + dset + '.csv', delimiter=',')
pY_GLP_RBF_cps=np.genfromtxt('./figures/pY_cps_GLP_RBF_' + dset + '.csv', delimiter=',')
pY_GLP_MAT_cps=np.genfromtxt('./figures/pY_cps_GLP_MAT_' + dset + '.csv', delimiter=',')
pY_TF_cps=np.genfromtxt('./figures/pY_cps_TF_' + dset + '.csv', delimiter=',')

CRPS_point_RF=np.genfromtxt('./figures/CRPS_point_RF_' + dset + '.csv', delimiter=',')
CRPS_point_GLP_RBF=np.genfromtxt('./figures/CRPS_point_GLP_RBF_' + dset + '.csv', delimiter=',')
CRPS_point_GLP_MAT=np.genfromtxt('./figures/CRPS_point_GLP_MAT_' + dset + '.csv', delimiter=',')
CRPS_point_TF=np.genfromtxt('./figures/CRPS_point_TF_' + dset + '.csv', delimiter=',')

CRPS_scc_RF=np.genfromtxt('./figures/CRPS_scc_RF_' + dset + '.csv', delimiter=',')
CRPS_ccc_RF=np.genfromtxt('./figures/CRPS_ccc_RF_' + dset + '.csv', delimiter=',')
CRPS_scc_GLP_RBF=np.genfromtxt('./figures/CRPS_scc_GLP_RBF_' + dset + '.csv', delimiter=',')
CRPS_ccc_GLP_RBF=np.genfromtxt('./figures/CRPS_ccc_GLP_RBF_' + dset + '.csv', delimiter=',')
CRPS_scc_GLP_MAT=np.genfromtxt('./figures/CRPS_scc_GLP_MAT_' + dset + '.csv', delimiter=',')
CRPS_ccc_GLP_MAT=np.genfromtxt('./figures/CRPS_ccc_GLP_MAT_' + dset + '.csv', delimiter=',')
CRPS_scc_TF=np.genfromtxt('./figures/CRPS_scc_TF_' + dset + '.csv', delimiter=',')
CRPS_ccc_TF=np.genfromtxt('./figures/CRPS_ccc_TF_' + dset + '.csv', delimiter=',')


# output values for table
if dset!='sl':
    
    minArray=np.zeros((4,5))

    minArray[0,0]=np.median(CRPS_point_RF[:,2])
    minArray[1,0]=np.median(CRPS_point_GLP_MAT[:,2])
    minArray[2,0]=np.median(CRPS_point_GLP_RBF[:,2])
    minArray[3,0]=np.median(CRPS_point_TF[:,2])

    minArray[0,1]=np.median(CRPS_pd_RF[:,2])
    minArray[1,1]=np.median(CRPS_pd_GLP_MAT[:,2])
    minArray[2,1]=np.median(CRPS_pd_GLP_RBF[:,2])
    minArray[3,1]=np.median(CRPS_pd_TF[:,2])

    minArray[0,2]=np.median(CRPS_cps_RF[:,2])
    minArray[1,2]=np.median(CRPS_cps_GLP_MAT[:,2])
    minArray[2,2]=np.median(CRPS_cps_GLP_RBF[:,2])
    minArray[3,2]=np.median(CRPS_cps_TF[:,2])

    minArray[0,3]=np.median(CRPS_scc_RF[:,2])
    minArray[1,3]=np.median(CRPS_scc_GLP_MAT[:,2])
    minArray[2,3]=np.median(CRPS_scc_GLP_RBF[:,2])
    minArray[3,3]=np.median(CRPS_scc_TF[:,2])


    np.savetxt('./figures/err_' + dset + '.csv', minArray, delimiter=",")


if dset=='sl':
    
    minArray=np.zeros((12,5))
    
    minArray[0,:]=np.median(CRPS_pd_RF,axis=0)
    minArray[1,:]=np.median(CRPS_pd_GLP_MAT,axis=0)
    minArray[2,:]=np.median(CRPS_pd_GLP_RBF,axis=0)
    minArray[3,:]=np.median(CRPS_pd_TF,axis=0)

    minArray[4,:]=np.median(CRPS_cps_RF,axis=0)
    minArray[5,:]=np.median(CRPS_cps_GLP_MAT,axis=0)
    minArray[6,:]=np.median(CRPS_cps_GLP_RBF,axis=0)
    minArray[7,:]=np.median(CRPS_cps_TF,axis=0)

    minArray[8,:]=np.median(CRPS_scc_RF,axis=0)
    minArray[9,:]=np.median(CRPS_scc_GLP_MAT,axis=0)
    minArray[10,:]=np.median(CRPS_scc_GLP_RBF,axis=0)
    minArray[11,:]=np.median(CRPS_scc_TF,axis=0)

    np.savetxt('./figures/err_splits_' + dset + '.csv', minArray, delimiter=",")  
#
