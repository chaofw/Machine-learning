#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chaofeng Wang

Convolutional NN using tensorflow with dropout
"""
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

####################################

tf.reset_default_graph() ## reset the default graph

mnist = input_data.read_data_sets('MINIST_data', one_hot = True)

############################
batch_size = 16
learning_rate = 0.01
MaxCount = 10000
num_class = 10
############################
##build CNN
input_layer = tf.placeholder(tf.float32, [None, 28*28], name = 'input_layer')
output_layer = tf.placeholder(tf.float32, [None, num_class], name = 'output_layer')
is_training = tf.placeholder(tf.bool, None)

##reshape input [batch_size, H * W] to [batch_size, H, W, channel] ## channel 3 for RGB and 1 for bw
image = tf.reshape(input_layer, [-1, 28, 28, 1])

## 1st convolution layer
conv1 = tf.layers.conv2d(inputs = image, filters = 8, kernel_size = 5, strides = 1, padding = 'same', activation = tf.nn.relu)

## filters: number of filters, i.e., number of neurons
## kernel_size: size of the window
## strides: moving step size of the filter
## padding: something similar to FFT to pad zeros after the end of signal
## output size [28, 28, filters]

## 1st pooling or subsampling layer
pool1 = tf.layers.max_pooling2d(conv1, pool_size = 2, strides = 2)
## output size [14, 14, filters]

## 2nd convolution layer
conv2 = tf.layers.conv2d(pool1, 10, 5, strides = 1, padding = 'same', activation = tf.nn.relu)
## output size [14, 14, 64]
pool2 = tf.layers.max_pooling2d(conv2, pool_size = 2, strides = 2)
## output size [7, 7, 64]


## Dense layer
pool2_flat = tf.reshape(pool2, [-1, 7*7*10]) ## flaten to a vector

dense_layer = tf.layers.dense(inputs = pool2_flat, units = 10, activation = tf.nn.relu)

## dropout
dropout = tf.layers.dropout(inputs = dense_layer, rate = 0.5, training = is_training)
## training = 1 then apply dropout ## = 0 then do not apply dropout for prediction

## output layer

final_output =  tf.layers.dense(dropout, num_class)
##end CNN##########################################

## define loss
loss = tf.losses.softmax_cross_entropy(onehot_labels = output_layer, logits = final_output) 

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()



################################################
##training
sess = tf.Session()

sess.run(init)

for ii in range(MaxCount):
    x,y = mnist.train.next_batch(batch_size)
    
    _, loss_ = sess.run([optimizer, loss], feed_dict = {input_layer: x, output_layer: y, is_training: 1})
    
    if ii % 50 == 0:
        print('loss:%0.4f' % loss_)

###################################################

##test

   
test_len = 200 ## number of test cases   


test_x = mnist.test.images[:test_len]
test_y = mnist.test.labels[:test_len]
            
test_y = np.argmax(test_y,1)

predict_y = sess.run(final_output, feed_dict = {input_layer: test_x, is_training: 0})

predict_ind = np.argmax(predict_y, 1)

accuracy = np.sum(test_y == predict_ind)/test_len

 
