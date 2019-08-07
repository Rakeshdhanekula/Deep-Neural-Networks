# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
#from keras import regularizers
#import cv2
# =============================================================================
# #from keras import backend as k

# from skimage.io import imread
# from skimage.transform import resize
# import numpy as np
#  
# =============================================================================
#cnn layer



   # resized_image = cv2.resize(image, (100, 50)) 
checkpoint_path_1 = "training_1/cp-{epoch:04d}.ckpt"
checkpoint_dir_1 = os.path.dirname(checkpoint_path_1)

checkpoint_path_2 = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir_2 = os.path.dirname(checkpoint_path_2)


# =============================================================================
# def classifier_1(loss1, unit):
#     adam = optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 10E-8, decay = 0.0, amsgrad = False)
# 
#     classifier = Sequential()
# 
#     classifier.add(Conv2D(16, (3, 3), input_shape = (128, 128, 3),  dilation_rate = (1, 1), activation = 'relu'))
#     classifier.add(MaxPooling2D(pool_size = (2, 2)))
# 
#     classifier.add(Conv2D(32, (3, 3), activation = 'relu' ))
#     classifier.add(MaxPooling2D(pool_size = (2, 2)))
# 
# 
# 
#     classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
#     classifier.add(MaxPooling2D(pool_size = (2, 2)))
# 
# 
# 
#     classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
#     classifier.add(MaxPooling2D(pool_size = (2, 2)))
# 
#     classifier.add(Flatten())
# 
#     classifier.add(Dense(units = 128, activation = 'relu', use_bias = True)) 
#  
#     classifier.add(Dropout(0.3))
#     classifier.add(Dense(units = unit, activation = 'sigmoid'))
# 
#     classifier.compile(optimizer = adam, loss = loss1, metrics = ['accuracy'])
#     
#     return classifier
# 
# =============================================================================


def classifier(loss, units):
    adam = optimizers.Adam(lr = 0.005, beta_1 = 0.9, beta_2 = 0.999, epsilon = 10E-8, decay = 0.0, amsgrad = False)

    classifier = Sequential()


    classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3),  activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))

    classifier.add(Conv2D(32, (3, 3), activation = 'relu',kernel_regularizer = regularizers.l2(0.01)))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    classifier.add(Conv2D(64, (3, 3), activation = 'relu',kernel_regularizer = regularizers.l2(0.01)))
    
    classifier.add(Conv2D(64, (3, 3), activation = 'relu',kernel_regularizer = regularizers.l2(0.01)))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    classifier.add(Conv2D(128, (3, 3), input_shape = (128, 128, 3),  activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))

    classifier.add(Conv2D(128, (3, 3), activation = 'relu',kernel_regularizer = regularizers.l2(0.01)))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    classifier.add(Conv2D(256, (3, 3), input_shape = (128, 128, 3),  activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))

    classifier.add(Conv2D(256, (3, 3), activation = 'relu',kernel_regularizer = regularizers.l2(0.01)))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    classifier.add(Flatten())

    classifier.add(Dense(units = 4096, activation = 'relu', use_bias = True, kernel_regularizer = regularizers.l2(0.01))) 
 
    classifier.add(Dropout(0.4))
    classifier.add(Dense(units = units, activation = 'sigmoid'))

    classifier.compile(optimizer = adam, loss = loss, metrics = ['accuracy'])
    
    return classifier





#fitting

train_datagenerator = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagenerator = ImageDataGenerator(rescale = 1./255)

training_set = train_datagenerator.flow_from_directory('data_set_1/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 8,
                                                 class_mode = 'binary')

test_set = test_datagenerator.flow_from_directory('data_set_1/test_set',
                                            target_size = (128, 128),
                                            batch_size = 8,
                                            class_mode = 'binary')
loss1 = 'binary_crossentropy'
model_1 = classifier(loss1, 1)
cp_callback_1 = tf.keras.callbacks.ModelCheckpoint(checkpoint_path_1, 
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=1)
model_1.fit_generator(training_set,
                      callbacks = [cp_callback_1],
                         steps_per_epoch = 800/8,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 200/8)
model_1.load_weights('/Users/rakeshdhanekula/Desktop/codes/Multi_prediction/training_1/cp-0004.ckpt')








train_datagenerator = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagenerator = ImageDataGenerator(rescale = 1./255)

training_set = train_datagenerator.flow_from_directory('data_set_2/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 10,
                                                 class_mode = 'categorical')

test_set = test_datagenerator.flow_from_directory('data_set_2/test_set',
                                            target_size = (128, 128),
                                            batch_size = 10,
                                            class_mode = 'categorical')
loss2 = 'categorical_crossentropy'
model_2 = classifier(loss2, 5)
cp_callback_2 = tf.keras.callbacks.ModelCheckpoint(checkpoint_path_2, 
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period =1)
                                                 
model_2.fit_generator(training_set,
                      callbacks = [cp_callback_2],
                         steps_per_epoch = 6000/10,
                         epochs = 50,
                         validation_data = test_set,
                         validation_steps = 4000/10)
#model_2.load_weights('/Users/rakeshdhanekula/Desktop/codes/Multi_prediction/training_2/cp-0008.ckpt')




