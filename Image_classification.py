#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 18:08:47 2019

@author: rakeshdhanekula
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


t = 100
i=0
training_data = []
direc = '/Users/rakeshdhanekula/Desktop/Master Thesis/data_set_for_class/training_set'
categories = ['class-1','class-2','class-3','class-4','class-5']
for category in categories:
    path = os.path.join(direc, category)
    class_num = categories.index(category) 
    for img in os.listdir(path):
        image_array = cv2.imread(os.path.join(path, img))
        training_data.append([image_array,class_num])
        
input_d = []
target = []        
cv_x = []
cv_y = []
test_x = []
test_y = []

for features, labels in training_data:
    input_d.append(features)
    target.append(labels)
    
v_x,test_x,v_y,test_y = train_test_split(input_d, target, test_size = 0.2)

X,cv_x,Y,cv_y = train_test_split(v_x,v_y, test_size = 0.6)
    
                
sess = tf.Session()        
    
    
epochs = 50    
nodes_fc1 = 4096
step_size = 10
steps = len(cv_x)
remaining = steps%step_size
validating_size = 40




x = tf.placeholder(tf.float32, shape = [None, 128, 128, 3])
y_true = tf.placeholder(tf.float32, shape = [None, 5])

w_1 = tf.Variable(tf.truncated_normal([7,7,3,64], stddev=0.01))
b_1 = tf.Variable(tf.constant(0.0, shape = [[5,5,3,64][3]]))

c_1 = tf.nn.conv2d(x, w_1, padding = 'VALID')
c_1 = c_1 + b_1
c_1 = tf.nn.relu(c_1)

p_1 = tf.nn.max_pool(c_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding= 'VALID')



w_2 = tf.Variable(tf.truncated_normal([5,5,64,128], stddev=0.01))
b_2 = tf.Variable(tf.constant(1.0, shape=[[5,5,64,128][3]]))

c_2 = tf.nn.conv2d(p_1, w_2, padding = 'VALID')
c_2 = c_2+b_2
c_2 = tf.nn.relu(c_2)

p_2 = tf.nn.max_pool(c_2, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'VALID')



w_3 = tf.Variable(tf.truncated_normal([3,3,128,256], stddev=0.01))
b_3 = tf.Variable(tf.constant(0.0, shape=[[3,3,128,256][3]]))

c_3 = tf.nn.conv2d(p_2,w_3, padding = 'VALID')
c_3 = c_3+b_3
c_3 = tf.nn.relu(c_3)

p_3 = tf.nn.max_pool(c_3, ksize=[1,2,2,1], strides= [1,2,2,1], padding = 'VALID')

flattened = tf.reshape(p_3, [-1,13*13*256])



input_size = int(flattened.get_shape()[1])

w1_fc = tf.Variable(tf.truncated_normal([input_size, nodes_fc1], stddev = 0.01))
b1_fc = tf.Variable(tf.constant(1.0, shape = [nodes_fc1]))

s_fc1 = tf.matmul(flattened, w1_fc)+b1_fc
s_fc1 = tf.nn.relu(s_fc1)

hold_prob1 = tf.placeholder(tf.float32)
s_fc1 = tf.nn.dropout(s_fc1, keep_prob = hold_prob1)



w2_fc = tf.Variable(tf.truncated_normal([nodes_fc1, 5], stddev = 0.01))
b2_fc = tf.Variable(tf.constant(1.0, shape = [5]))

y_pred = tf.matmul(s_fc1, w2_fc)
print(y_pred)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_true, logits = y_pred))
cost = tf.reduce_mean(cross_entropy)

train = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(cross_entropy)



matches = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true,1))
acc = tf.reduce_mean(tf.cast(matches, tf.float32))


init = tf.global_variables_initializer()



acc_list = []
auc_list = []
loss_list = []
saver = tf.train.Saver()

config = tf.ConfigProto(device_count = {'GPU': 0})
with tf.Session(config=config) as sess:
    sess.run(init)
    for i in range(epochs):
        for j in range(0,steps-remaining,step_size):
            #Feeding step_size-amount data with 0.5 keeping probabilities on DROPOUT LAYERS
            _,c = sess.run([train,cross_entropy],
			feed_dict={x:X[j:j+step_size] ,y_true:Y[j:j+step_size] ,hold_prob1:0.5})
        
        
		#Writing for loop to calculate test statistics. GTX 1050 isn't able to calculate all test data.
        cv_auc_list = []
        cv_acc_list = []
        cv_loss_list = []
        for v in range(0,len(cv_x)-int(len(cv_x) % validating_size),validating_size):
            acc_on_cv,loss_on_cv,preds = sess.run([acc,cross_entropy,tf.nn.softmax(y_pred)],
			feed_dict={x:cv_x[v:v+validating_size] ,y_true:cv_y[v:v+validating_size] ,hold_prob1:1.0})
			
            auc_on_cv = roc_auc_score(cv_y[v:v+validating_size],preds)
            cv_acc_list.append(acc_on_cv)
            cv_auc_list.append(auc_on_cv)
            cv_loss_list.append(loss_on_cv)
        acc_cv_ = round(np.mean(cv_acc_list),5)
        auc_cv_ = round(np.mean(cv_auc_list),5)
        loss_cv_ = round(np.mean(cv_loss_list),5)
        acc_list.append(acc_cv_)
        auc_list.append(auc_cv_)
        loss_list.append(loss_cv_)
        print("Epoch:",i,"Accuracy:",acc_cv_,"Loss:",loss_cv_ ,"AUC:",auc_cv_)
    
    test_auc_list = []
    test_acc_list = []
    test_loss_list = []
    for v in range(0,len(test_x)-int(len(test_x) % validating_size),validating_size):
        acc_on_test,loss_on_test,preds = sess.run([acc,cross_entropy,tf.nn.softmax(y_pred)], 
		feed_dict={x:test_x[v:v+validating_size] ,
		y_true:test_y[v:v+validating_size] ,
		hold_prob1:1.0})
        
        auc_on_test = roc_auc_score(test_y[v:v+validating_size],preds)
        test_acc_list.append(acc_on_test)
        test_auc_list.append(auc_on_test)
        test_loss_list.append(loss_on_test)
    saver.save(sess, os.path.join(os.getcwd(),"CNN_MC.ckpt"))
    test_acc_ = round(np.mean(test_acc_list),5)
    test_auc_ = round(np.mean(test_auc_list),5)
    test_loss_ = round(np.mean(test_loss_list),5)
    print("Test Results are below:")
    print("Accuracy:",test_acc_,"Loss:",test_loss_,"AUC:",test_auc_)








































































