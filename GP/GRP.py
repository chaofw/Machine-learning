#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chaofeng Wang

Gaussian Process Regression for scalar feature
"""
import numpy as np
import matplotlib.pyplot as plt
import pylab

########################################
class GPR:
    
    def __init__(self, obv , obv_x, sigma_f, L, query):
        
        self.obv = obv
        self.obv_x = obv_x
        
        self.N = obv.shape[0]
        
        self.sigma_f = sigma_f
        
        self.L = L
        
        self.query = query
        
        
    def regression(self):
        
        Kx = self.CovMat(self.obv_x, self.obv_x)
        
        Kq = self.CovMat(self.query, self.query)
        
        Kqx = self.CovMat(self.query, self.obv_x)
        
        self.mu = Kqx @ np.linalg.inv(Kx) @ obv  ## assuming prior mean as 0
        
        self.cov = Kq - Kqx @ np.linalg.inv(Kx) @ Kqx.T
        
            
            
    def CovMat(self,x,y):
        
        K = np.zeros((x.shape[0],y.shape[0]))
        
        for ii in range(x.shape[0]):
            
            for jj in range(y.shape[0]):
                
                K[ii,jj] = self.CovFun(x[ii],y[jj])
        return K
        
    
    def CovFun(self,x1,x2):  # covariance function
        
        temp = (x1 - x2) ## must in matrix form
        r = temp * self.L * temp
        
        return self.sigma_f * (1 + np.sqrt(3 * r)) * np.exp(-np.sqrt(3 * r))
    


########################################
    

    
    
