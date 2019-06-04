
# bash  hw8_test.sh  <testing data>  <prediction file>

import time, os, sys

import numpy as np, pandas as pd; np.random.seed(0)
import random as rn; rn.seed(12345)

import tensorflow as tf
from tensorflow.python.keras import backend as K

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from tensorflow.python.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.python.keras.layers import Conv2D, Activation
from tensorflow.python.keras.layers import BatchNormalization, Reshape
from tensorflow.python.keras.layers import DepthwiseConv2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.models import Model, Sequential

from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.optimizers import Adam


from pandas import read_csv
from csv import writer

tag_test, x_test = read_csv(sys.argv[1]).to_numpy().T

total_test = len(x_test)
for i in range(total_test):
    print('\r', '%7.4lf'%(i / total_test * 100), '%', end='\r')
    x_test[i] = np.array([int(pixel) 
        for pixel in x_test[i].split()]).reshape(48, 48, 1) 
x_test = np.stack(x_test)
x_test = x_test / 255.

model = Sequential([
    BatchNormalization(input_shape=(48, 48, 1)),
    Conv2D(24, 5), Activation('relu'), 
    Conv2D(36, 5), Activation('relu'), 
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    BatchNormalization(),
    Conv2D(52, 1), Activation('relu'), 
    DepthwiseConv2D(5, padding='same'), 
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(rate=0.4),
    
    BatchNormalization(),
    Conv2D(64, 1), Activation('relu'), 
    DepthwiseConv2D(5, padding='same'), 
    DepthwiseConv2D(5, padding='same'), 
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    BatchNormalization(),
    Conv2D(84, 1), Activation('relu'), 
    DepthwiseConv2D(5, padding='same'), 
    DepthwiseConv2D(5, padding='same'), 
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(rate=0.4),
    
    Flatten(),
    
    Dense(90, activation='relu'), Dropout(rate=0.5),
    Dense(7, activation='softmax')
])

model.set_weights(np.load(sys.argv[3], allow_pickle=True))
ans = model.predict(x_test).argmax(1)
with open(sys.argv[2], 'w') as f:
    csvwriter = writer(f)
    f.write('id,label\n')
    csvwriter.writerows(list(enumerate(ans)))

K.clear_session()