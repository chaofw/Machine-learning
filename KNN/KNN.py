#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN

@author: Chaofeng Wang

example is from the internet
"""

import numpy as np

###################################
class KNN:
    
    
    def __init__(self,K,data_x,data_y,x):
        
        self.K = K ## number of neighbours
        self.data_x = data_x
        self.data_y = data_y
        self.x = x
        
    def train(self):
        
        ##calculate distance
        
        D = np.zeros(len(self.data_y))
        
        for ii in range(len(self.data_y)):
            
            D[ii] = self.distance(self.data_x[ii,:],self.x)
            
        index = np.argsort(D)
        
        vote_total = {} ## hash table
        
        for ii in range(self.K):
            
            vote = self.data_y[index[ii]]
            
            ## better way to intert element in hash table
            """
            vote_total[vote] = vote_total.get(vote,0) + 1
            """
            
            if vote not in vote_total:
                vote_total[vote] = 0.0
            else:
                vote_total[vote] = vote_total[vote] + 1.0
                
                
        ## find the maximum from the hash table        
        Max = max(vote_total, key=lambda key: vote_total[key])
        ##    
        self.label = Max
            
        
    def distance(self,x,y):
        
        Dist = np.linalg.norm(x - y) ## Euclidean distance ## can be changed
        
        return Dist
        


#######################################

if __name__ == "__main__":
    
    
    ##test data from example in Google
    data_x = np.array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    data_y = np.array([1, 1, 0, 0])
    ####
    
    x = np.array([0.1, 0.3]) 
    
    K = 3
    
    a = KNN(K,data_x,data_y,x)
    
    a.train()
    
    
