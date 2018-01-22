import numpy as np
########################################################################
class Solution:
    """
    @x is feature
    @y is target value
    @x,y are np.matrix
    @alpha is learning rate
    """
    def LR(self,x,y): ## LS solution
        # write your code here
        xsize = x.shape # assume one column
        #ysize = y.shape
        
        X = np.hstack((x,np.ones((xsize[0],1))))
        
 
        ifinv = np.linalg.inv((X.transpose() * X))

        
        
        Beta =  ifinv * X.transpose() * y
        
        return Beta
        
    def LR_batch(self,x,y,alpha): ## batch gradient descent

        xsize = x.shape     
        
        if len(xsize) == 1:
            error = np.ones((1,1))
        else:
            error = np.ones((xsize[1] + 1,1))

        Beta = error        
        
        X = np.hstack((x,np.ones((xsize[0],1))))
        
        count = 0

        Maxcount = 200       

        #alpha = 0.2      
        
        while np.linalg.norm(error) > 0.001 and count < Maxcount:
            Beta = Beta - alpha * X.transpose() * (X * Beta - y)
            count = count + 1
            
            error = alpha * X.transpose() * (X * Beta - y)
            
        return Beta
        
    def LR_SGD(self,x,y,alpha):
        
        xsize = x.shape
        
        if len(xsize) == 1:
            error = np.ones((1,1))
        else:
            error = np.ones((xsize[1] + 1,1))
            
        Beta = error
        
        #Beta = Beta
        
        X = np.hstack((x,np.ones((xsize[0],1))))
        
        count = 0
        
        Maxcount = 200
        
        while np.linalg.norm(error) > 0.001 and count < Maxcount:
            
            Beta_ini = Beta            
            
            for ii in range(xsize[0]):
            
                Beta = Beta - alpha * X[ii,:].transpose() * (X[ii,:] * Beta - y[ii,0])
                
                
            count = count + 1
            
            error = Beta - Beta_ini
        return Beta
        
        
        
        
        
        
 ###########################################################
        
if __name__ == "__main__":

    a = Solution()

    x = np.matrix(input)

    y = np.matrix(output)

    beta = a.LR(x,y)

    alpha = 0.2

    beta1 = a.LR_batch(x,y,alpha)   


    beta2 = a.LR_SGD(x,y,alpha)
