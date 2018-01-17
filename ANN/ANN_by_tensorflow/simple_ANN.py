#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chaofeng Wang

set simple ANN by TensorFlow

"""

import tensorflow as tf
import numpy as np

class ANN_TF:
    
    def __init__(self,strcture,session):
        
        ##
        self.num_input = structure[0]
        self.num_output = structure[-1]
        self.num_hid = len(structure) - 2
        self.num_hid_node = structure[1:-1]
        ##
        self.hidden_layer = []
        ##
        self.session = session
        
    def build_hid_layer(self, out_pre, n_in, n_out, activation):
        
        weights = tf.Variable(tf.random_normal([n_out, n_in]))
        bias = tf.Variable(tf.random_normal([n_out, 1]))  ## need initialization
        
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights)
        
        out_cur = tf.matmul(weights,out_pre) + bias
        
        return activation(out_cur)
        
    def build_ANN(self,x,y,learning_r,C):
        
        self.feature = x
        self.label = y
        self.learning_r = learning_r
        self.C = C
        
        ## build input and output layer
        self.input_layer = tf.placeholder(tf.float32,shape = [self.num_input, None]) ## None means arbitrary number
        self.label_layer = tf.placeholder(tf.float32,shape = [self.num_output, None]) ## do not need initialization
        ##
        
        
        ##build hidden layer
        ##first hidden layer
        self.hidden_layer.append(self.build_hid_layer(self.input_layer, self.num_input, self.num_hid_node[0], activation = tf.nn.sigmoid))
        ## other hidden layers
        for ii in range(self.num_hid - 1):
            
            self.hidden_layer.append(self.build_hid_layer(self.hidden_layer[-1], self.num_hid_node[ii],self.num_hid_node[ii+1], activation = tf.nn.sigmoid))
        
        
        self.output_layer = self.build_hid_layer(self.hidden_layer[-1],self.num_hid_node[-1],self.num_output,activation = tf.nn.sigmoid)
        
        
        
        
    def train(self, MaxCount):
        
        self.loss = tf.reduce_mean(tf.square((self.output_layer - self.label_layer)),[0,1])
        
        ## adding regularization
        regularizer = tf.contrib.layers.l2_regularizer(scale = self.C)
        
        tf.contrib.layers.apply_regularization(regularizer)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        
        self.loss = self.loss + tf.reduce_sum(reg_losses)

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_r).minimize(self.loss)
        
        init = tf.global_variables_initializer() ## initialize variables
        
        self.session.run(init)
        
        ###iteration
        
        for ii in range(MaxCount):
            self.session.run(self.optimizer, feed_dict = {self.input_layer: self.feature, self.label_layer: self.label})
        
        
        
    
    def predict(self, feature):
        
        return self.session.run(self.output_layer, feed_dict = {self.input_layer: feature})
        
        
        
####################################
#end class definition
        
if __name__ == "__main__":
        
    tf.reset_default_graph()


    cases = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    
    labels = [[0], [1], [1], [0]]
    
    x = np.matrix(cases).T
    
    y = np.matrix(labels).T
    
    learning_r = 0.1 # learning rate
    C = 0.0001 # weight decay
    
    structure = [2,10,5,1] ## number of nodes in each layer
    
    y_test = []
    
    with tf.Session() as session:
        ANN = ANN_TF(structure,session)
        ANN.build_ANN(x,y,learning_r,C)
        ANN.train(10000)
        
        #writer = tf.summary.FileWriter("logs/", session.graph) ## save the graph

        
        for ii in range(4):
            y_test.append(ANN.predict(x[:,ii]))
            print(y_test[-1])
            
        """ ####check the weights
        variables_names =[v.name for v in tf.trainable_variables()]
        values = session.run(variables_names)
        for k,v in zip(variables_names, values):
            print(k, v)
        """
    
      
