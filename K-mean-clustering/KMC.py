#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K mean clustering

@author: Chaofeng

simple K mean clustering for two Gaussians

"""

import numpy as np

###################################################
class KMC:
    
    def __init__(self, K, data, init_center):
        
        self.data = data[:]
        
        self.center = init_center[:] ## assign an array to another array
        
        self.K = K
        
        self.N = data.shape[0] ## number of data samples
        
        self.M = data.shape[1] ## dim of feature
        
    def distance(self,x,y):
        
        Dist = np.linalg.norm(x - y)
        
        return Dist
        
        
    def train(self):
        
        center_pos = np.copy(self.center)
        
        center_pri = np.zeros((self.K,self.M))
        
        clusInd = np.zeros(self.N)
        
        tol = 0.001
        
        count = 0
        
        MaxCount = 10000
        
        while np.linalg.norm(center_pri - center_pos) > tol and count < MaxCount:
                        
            center_pri = np.copy(center_pos) ### PAY ATTENTION TO THIS!!!
            
            count = count + 1
            
            center_count = np.zeros(self.K)
            
            for ii in range(self.N): # iteration for all the samples
                
                xtemp = self.data[ii,:] ## one sample
                
                minDict = np.inf  ## min distance
                
                
                for jj in range(self.K): ## iteration for all clusters
                    
                    dist_temp = self.distance(xtemp, center_pri[jj,:])
                    
                    if dist_temp < minDict:
                        
                        minDict = dist_temp
                        
                        clusInd[ii] = jj
                        
                index_temp = int(clusInd[ii])
                        
                center_pos[index_temp,:] = center_pos[index_temp,:] + xtemp ## update the center
  
                center_count[index_temp] = center_count[index_temp] + 1.0 ## number of sample for each cluster
                ### end for jj
                
            
            ### end of ii
            
            for ii in range(self.K):
                
                center_pos[ii,:] = center_pos[ii,:] / center_count[ii]
                
            
                
        ###end for while                
        
        self.center = center_pos
        
        self.label = clusInd
        
        self.count = count
            
                        
        
        
        


###################################################


if __name__ == "__main__":
    
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[0.8, 0.6], [0.6, 0.8]])
    X1 = np.random.multivariate_normal(mean1, cov, 100)
    X2 = np.random.multivariate_normal(mean2, cov, 100)
    
    
    X = np.concatenate((X1,X2))
    
    ###########################3
    
    
    K = 2
    
    index = np.arange(100)
    
    np.random.shuffle(index) 
    
    init_center = np.zeros((K,2))
    
    for ii in range(K):
        
        init_center[ii] = X[index[ii],:] 
    
    a = KMC(K,X,init_center)
    
    a.train()
    
