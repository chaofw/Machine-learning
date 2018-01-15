# -*- coding: utf-8 -*-
"""

Author: Chaofeng Wang

Logistic regression

Implemented gradient ascent by batch and mini batch version

very sensitive to outliers

"""
import numpy as np

class Logi_regression:
    
    def LoR_BGA(self,x,y,alpha):
	
        xsize = x.shape

        error = np.matlib.ones((xsize[0],1)) # initialize error vector

        coef = np.matlib.zeros((xsize[1]+1,1)) # initialize coefficients

        X = np.hstack((x,np.ones((xsize[0],1)))) # adding intercept

        maxCount = 2000

        count = 0

        while np.linalg.norm(error) > 0.001 and count < maxCount:
		
                count = count + 1
		
                error = y - (1 / (1 + np.exp(-X * coef)))

                coef = coef + alpha * X.T  * error
		
        return coef



    def LoR_SGA(self,x,y,alpha):
        xsize = x.shape

        error = np.matlib.ones((xsize[1],1)) # initialize error vector

        coef = np.matlib.zeros((xsize[1]+1,1)) # initialize coefficients

        X = np.hstack((x,np.ones((xsize[0],1)))) # adding intercept

        maxCount = 2000

        count = 0
        
        B = 2 # mini-batch size
        
        index_total = range(xsize[0])

        while np.linalg.norm(error) > 0.001 and count < maxCount:
                
                index = np.random.choice(index_total,B,replace = False) # random choose index
            
            
                y_batch = y[index,:]
                
                X_batch = X[index,:]
                
                count = count + 1
		
                error = y_batch - (1 / (1 + np.exp(-X_batch * coef)))

                coef = coef + alpha * X_batch.T  * error
		
        return coef




if __name__ == "__main__":
    
    a = Logi_regression()
    ### input data as x,y
    x = np.matrix(x)
    y = np.matrix(y)

    x = x.T # column vector
    y = y.T #column vector

    alpha = 0.001 # learning rate

    coef_BGA = a.LoR_BGA(x,y,alpha)
    coef_SGA = a.LoR_SGA(x,y,alpha)
    
    xsize = x.shape
    X = np.hstack((x,np.ones((xsize[0],1))))

    r_BGA_err =  y- (1 / (1 + np.exp(-X * coef_BGA)))
    r_SGA_err =  y- (1 / (1 + np.exp(-X * coef_SGA)))

    
