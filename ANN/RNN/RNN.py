#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chaofeng Wang

RNN

exercise coding based on online sources for MNIST
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#########################################################################
class RNN:
    
    def __init__(self, session, B, time_step, input_size, learning_rate, num_hid):
        
        self.session = session
        self.B = B ## batch size
        self.time_step = time_step
        self.input_size = input_size
        
        self.learning_rate = learning_rate
        self.num_hid = num_hid ## number of hidden layer in LSTM
        
        
    def train_RNN(self, output_size, MaxCount, data): ## assuming one hidden layer


        ## input and output
        self.input_layer = tf.placeholder(tf.float32, shape = [None, self.time_step * self.input_size],name = "input_layer")
        
        output_layer = tf.placeholder(tf.int32, shape = [None, output_size])
        
        ### get RNN
            
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.num_hid)  ## 
        
        x = tf.reshape(self.input_layer, [-1, self.time_step, self.input_size]) ## (batch, size of image) to (batch, time_step, input_size)
        
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, initial_state = None, dtype = tf.float32, time_major = False)
        
        ## time major = false for (batch,...) otherwise (time step, batch,..)
        
        final_output = tf.layers.dense(outputs[:,-1,:],output_size)  ## use the last output of the network
        
        final_output = tf.identity(final_output, name="final_output") ## name the output for prediction
        
        
        loss = tf.losses.softmax_cross_entropy(logits = final_output, onehot_labels = output_layer)
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(loss)
        
        init = tf.global_variables_initializer() ## initialize variables
        
        self.session.run(init)
        
        ###evaluation
        
        
        
        for ii in range(MaxCount):
            x,y = data.train.next_batch(self.B)
            
            _, loss_ = self.session.run([optimizer, loss], feed_dict = {self.input_layer: x, output_layer: y})
            
            if ii % 50 == 0:
                print('train loss:%.4f' % loss_)
        
        
    def predict(self, final_output, feature):
        
        index = self.session.run(final_output, feed_dict = {self.input_layer: feature})
        
        return np.argmax(index, 1)
        
        
        
#########################################################################


if __name__ == "__main__":
    
        tf.reset_default_graph()
        
        mnist = input_data.read_data_sets('MINIST_data', one_hot = True)
        
        batch_size = 32
        
        num_hid = 32
        
        test_len = 200
        
        with tf.Session() as sess:
            
            RNN = RNN(sess, batch_size, 28, 28, 0.01, num_hid)
            
            RNN.train_RNN(10,10000,mnist)
            
            
            ############test
            
            test_x = mnist.test.images[:test_len]
            test_y = mnist.test.labels[:test_len]
            
            test_y = np.argmax(test_y,1) ## int
            
            final_output = sess.graph.get_tensor_by_name("final_output:0")
            
            predict_y = RNN.predict(final_output, test_x) ## int
            
            accuracy = np.sum(test_y == predict_y)/test_len
            
            saver = tf.train.Saver()
            
            save_path = saver.save(sess, "RNN_Model")
            
            
            
            
            writer = tf.summary.FileWriter("logs/", sess.graph)
        

    
