#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chaofeng Wang

RNN

exercise coding based on online sources for MNIST classification
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#########################################################################
class RNN:
    
    def __init__(self, session, B, time_step, input_size, learning_rate):
        
        self.session = session
        self.B = B
        self.time_step = time_step
        self.input_size = input_size
        
        self.learning_rate = learning_rate
        
        
        
    def built_ANN(self, output_size, activation): ## assuming one hidden layer

        self.input_layer = tf.placeholder(tf.float32, shape = [None, self.time_step * self.input_size])
        
        self.input_layer = tf.placeholder(tf.int32, shape = [None, output_size])
        
        
        
        
#########################################################################


if __name__ == "__main__":
    
        tf.reset_default_graph()
        
        minist = input_data.read_data_sets('MINIST_data', one_hot = True)
        
        z = minist.test.image
        
        

    
