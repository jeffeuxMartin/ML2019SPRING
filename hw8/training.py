import time, os, sys

import numpy as np, pandas as pd

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

y_train, x_train = read_csv(sys.argv[1]).to_numpy().T

total_train = len(x_train)
for i in range(total_train):
    print('\r', '%7.4lf'%(i / total_train * 100), '%', end='\r')
    x_train[i] = np.array([int(pixel) 
        for pixel in x_train[i].split()]).reshape(48, 48, 1) 
x_train = np.stack(x_train)
x_train = x_train / 255.
y_train = to_categorical(y_train)


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


import time
def now_name():
    return time.strftime("%m%d_%H%M%S", time.gmtime(time.time() + 8*60*60)) 
now_n = now_name()


from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import EarlyStopping

train_x, val_x = x_train[:-2871], x_train[-2871:]
train_y, val_y = y_train[:-2871], y_train[-2871:]

datagen = ImageDataGenerator(
    horizontal_flip=True,
    zoom_range=0.1,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1
    )
datagen.fit(train_x)



checkpointer = ModelCheckpoint(
    filepath=(sys.argv[2]),
    monitor='val_acc', verbose=1, save_best_only=True,
    save_weights_only=True)
print('Model saved at: << ', sys.argv[2], ' >>.')
model.save_weights(sys.argv[2])

early_stopping = EarlyStopping(
    monitor='val_loss', patience=50, verbose=1,
    restore_best_weights=False)

adam = Adam()
model.compile(adam, 'categorical_crossentropy', metrics=['acc'])
model.fit_generator(
    datagen.flow(train_x, y=train_y, batch_size=96),
    epochs=200,
    validation_data=(val_x, val_y),
    callbacks=[checkpointer, early_stopping])

model.load_weights(sys.argv[2])

W1 = np.array([_wei.astype(np.float16) for _wei in model.get_weights()])
ws = np.array([w.shape for w in W1])
wt = np.concatenate([w.reshape(-1) for w in W1], 0)

np.savez_compressed(sys.argv[2].split('.')[0], wt=wt, ws=ws)