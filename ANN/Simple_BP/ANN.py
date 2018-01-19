#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chaofeng Wang

MLP: simple BP algorithm


"""

import numpy as np


##########################################3333

class ANN:

    def __init__(self,x,y,num_hid,num_node,C,learning_r):
        
        self.num_input = x.shape ## [number of feature,number of samples]
        
        self.num_output = y.shape ## [number of label, number of samples]
        
        self.intercept_list = []
        
        self.weight_list = []
        
        self.num_hid = num_hid
        
        self.num_node = num_node
        
        self.learning_r = learning_r
        
        self.C = C
        
        
        epsilon = 1
        
        for ii in range(num_hid + 1):
            self.intercept_list.append(np.random.normal(0,epsilon,(num_node[ii+1],1)))
            self.weight_list.append(np.matrix(np.random.normal(0,epsilon,(num_node[ii+1],num_node[ii]))))
            
            #self.intercept_list.append(np.matrix(np.ones((num_node[ii+1],1))))
            #self.weight_list.append(np.matrix(np.ones((num_node[ii+1],num_node[ii]))))
            
        

    
    def BP_train(self,x,y):
        
        
        ## feature vector x
        ## label vector y
        ## number of hidden layer num_hid
        ## number of nodes in each hidden layer as a vector num_node
        ## weight decay factor preventing overfit C
        
        tol = 0.01 # tolerence for stopping
        MaxCount = 10000 # maximum number of iteration
        
        count = 0
        
        error = 100
        
        ###################################
        while count < MaxCount and error > tol: ## iteration
            
            
            w_temp = self.weight_list
                
            b_temp = self.intercept_list
            
            
            for ii in range(self.num_input[1]): ## enumerate samples
            
                x_temp = x[:,ii]  ## each sample
                
                y_temp = y[:,ii]
                
               
                ## forward process
                
                Forward = self.Predict(x_temp)
                
                ## Forward  [output, net]
                
                ## output
                
                output = Forward[0]
                
                net = Forward[1]
                
                ###output layer
                delta = np.multiply((output[-1] - y_temp),self.act_der(net[-1]))
                
                b_temp[-1] = b_temp[-1] - self.learning_r/self.num_input[1] * delta
                
                w_temp[-1] = w_temp[-1] - self.learning_r/self.num_input[1] * delta * output[-2].T
                
                w_temp[-1] = w_temp[-1] - self.learning_r*self.C/self.num_input[1] * self.weight_list[-1]

                #######
                
                for jj in range(self.num_hid):
                    
                    delta = np.multiply(self.weight_list[self.num_hid - jj].T * delta,self.act_der(net[self.num_hid -jj - 1]))
            
                    
                    b_temp[self.num_hid - jj - 1] = b_temp[self.num_hid - jj - 1] - self.learning_r/self.num_input[1] * delta
                    
                    w_temp[self.num_hid - jj - 1] = w_temp[self.num_hid - jj - 1] - self.learning_r/self.num_input[1] * delta * output[self.num_hid -jj - 1].T
                    
                    
                    w_temp[self.num_hid - jj - 1] = w_temp[self.num_hid - jj - 1] - self.learning_r*self.C/self.num_input[1] * self.weight_list[self.num_hid - jj -1]
            ##end for one sample
#                self.weight_list = w_temp
#                self.intercept_list = b_temp
#                Forward = self.Predict(x_temp)
            ####calculate error
            
            self.weight_list = w_temp
            
            self.intercept_list = b_temp
            
            error = 0.0
            
            for ii in range(self.num_input[1]):
                
                x_temp = x[:,ii]  ## each sample
                
                y_temp = y[:,ii]
                
               
                ## forward process
                
                Forward = self.Predict(x_temp)
                
                error = error + np.linalg.norm((Forward[0][-1]-y_temp))
                
            
            
            count = count + 1 ## end of iteration    
        ######################################  end while
        self.error = error
        self.count = count
        
        ###end BP

            
        
    def Predict(self,x):
        
        # predict
        
        output = []
        
        output.append(x)
        
        net = []
        
        for ii in range(self.num_hid + 1):
            
            temp = output[ii]
            
            net.append(self.weight_list[ii] * temp + self.intercept_list[ii])
            
            output.append(self.sigmoid_fun(net[ii]))
            
        return output,net
    
    def sigmoid_fun(self,x):
        
        return 1.0 / (1.0 + np.exp(-x)) 
    
    def act_der(self,x): ## derivation of activation function
    
        return np.multiply((1 - self.sigmoid_fun(x)),self.sigmoid_fun(x))
        


##############################################




if __name__ == "__main__":
    
    N = 20
    
    
    cases = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    
    labels = [[0], [1], [1], [0]]
    
    x = np.matrix(cases).T
    
    y = np.matrix(labels).T
    
    #######################################
    learning_r = 0.8 # learning rate, can be determined by grid search
    C = 0 # penalty, can be determined by grid search
    #######################################
    
    num_hid =1
    
    num_node = [2,6,1];
            
    a = ANN(x,y,num_hid,num_node,C,learning_r)
    
    a.BP_train(x,y)   
    
    yp = []
    
    for ii in range(4):

        yp.append(a.Predict(x[:,ii])[0][-1])
    
