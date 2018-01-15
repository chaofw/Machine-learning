"""

Author: Chaofeng Wang

SVM with QP (Alternative: use SMO)

Dual form:

"""

import numpy as np
import cvxopt
import cvxopt.solvers

######################################

class SVM:
    
    def SVM_QP(self,x,y,C): # SVM by QP
        
        N = x.shape[0]
        
        K = self.Kernal_matrix(x)
        
        P = cvxopt.matrix(np.diag(y.A1)*K*np.diag(y.A1))
        
        qq = cvxopt.matrix(-np.ones(N))
        
        A = cvxopt.matrix(y).T
        
        bb = cvxopt.matrix(0.0)# not intercept
        
        G1 = -cvxopt.matrix(np.eye(N)) # greater than zero
        
        h1 = cvxopt.matrix(np.zeros(N))
        
        G2 =  cvxopt.matrix(np.eye(N)) # box constraint
        
        h2 = C * cvxopt.matrix(np.ones(N))
        
        G_all = cvxopt.matrix(np.vstack((G1,G2)))
        
        h_all = cvxopt.matrix(np.vstack((h1,h2)))
        
        cvxopt.solvers.options['show_progress'] = False
        
        solution = cvxopt.solvers.qp(P, qq, G_all, h_all, A, bb)
        
        ##find the support vectors####################################
        alpha = np.ravel(solution['x'])
        
        judge = alpha > 1e-5
        
        index = np.arange(len(alpha))[judge]
        
        
        self.alpha_sv = np.matrix(alpha[index]).T
        
        self.x_sv = np.matrix(x[index])
        
        self.y_sv = np.matrix(y[index])
        
        
        ##find the intercept
        
        self.b = 0.0
        self.temp = np.multiply(self.alpha_sv,self.y_sv)
        
        for ii in range(len(index)):
            
            self.b += self.y_sv[ii] - self.temp.T * K[index,index[ii]]

        self.b = self.b/len(index)
        
    def predict(self,x):
        
        y_p = np.matrix(np.zeros(len(x)))
        
        
        for ii in range(len(x)):
            
            for jj in range(len(self.alpha_sv)):
                y_p[0,ii] += self.alpha_sv[jj] * self.y_sv[jj] * self.Kernal(x[ii],self.x_sv[jj])
         
  
        return y_p + self.b
        
    #def SVM_SMO(self,x,y,C): # SVM by SMO
        
        
    def Kernal(self,x1,x2):  # calculate Kernal function, can be modified by applications
                    # x is a column vector
        #linear kernal
        
        return x1*x2.T### linear kernal
        

    def Kernal_matrix(self,x): ## calculate kernal matrix
        
        N = x.shape[0]
        
        K = np.matrix(np.zeros((N, N)))
        
        for ii in range(N):
            for jj in range(N):
                K[ii,jj] = self.Kernal(x[ii],x[jj])
        
        return K
         
        
        
#############################################
## test case
        
if __name__ == "__main__":
    
    a = SVM()
    
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[0.8, 0.6], [0.6, 0.8]])
    X1 = np.random.multivariate_normal(mean1, cov, 100)
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 100)
    y2 = np.ones(len(X2)) * -1
    
    x = np.matrix(np.vstack((X1,X2)))
    
    y = np.matrix(np.hstack((y1,y2))).T
    
    
    C = 100 # penalty, can be determined by grid search
            # infinity as hard margin (np.inf)
            
    a.SVM_QP(x,y,C)
    
    Y = a.predict(x).T
            
    
    

