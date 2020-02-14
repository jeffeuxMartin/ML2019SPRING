import time
import os
import sys
import argparse
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import PReLU
from keras.optimizers import RMSprop, Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as backend

def now_name():
    return time.strftime(
        "%m%d_%H%M%S", 
        time.gmtime(time.time() + 8 * 60 * 60)) 

def parsing_arguments():
    parser = argparse.ArgumentParser(
        description='Process data filenames.')
    parser.add_argument('--trainage', '-tr', '--tr',
                        dest='tr', help='', default='')
    # parser.add_argument('--testage', '-te', '--te',
    #                     dest='te', help='',
    #                     default='Trainpack/Colab/'\
    #                             'ML HW3/data/test.csv')
    parser.add_argument('--weight', '-wt', '--wt', 
                        dest='wt', help='', default='.')
    # parser.add_argument('--results', '-rs', '--rs',
    #                     dest='rs', help='', 
    #                     default='Trainpack/Colab/'\
    #                             'ML HW3/results/')
    return parser.parse_args()

backend.tensorflow_backend._get_available_gpus()

args = parsing_arguments()
train_path = args.tr
# test_path = args.te
wght_path = args.wt
# rslt_path = args.rs
if train_path == "":
	print("Please input your training data!")

batch_size = 105
num_classes = 7
epochs = 250
training_perc = 10

# flip_aug = True

train_data = [[np.fromstring(entry[1], sep=' '), entry[0]] \
               for entry in pd.read_csv(train_path).values]
# test_data = [[np.fromstring(entry[1], sep=' '), entry[0]]\
#                for entry in pd.read_csv(test_path).values]

x_train, y_train = np.array(train_data).T
x_train = np.concatenate(x_train).reshape(-1, 48, 48, 1)\
                                    .astype('float32') / 255
x_train, y_train = np.concatenate(
                            [x_train[:59], x_train[60:]]), \
                   np.concatenate(
                            [y_train[:59], y_train[60:]])

# x_test, x_test_id = np.array(test_data).T
# x_test = np.concatenate(x_test).reshape(-1, 48, 48, 1)\
#                                 .astype('float32') / 255

# Feature scaling???
# num = x_train.shape[0]
# print('...', num, 'samples loaded.')
training_part = int(num * (1 - training_perc / 100))
x_train, x_val = \
            x_train[:training_part], x_train[training_part:]
y_train, y_val = \
            y_train[:training_part], y_train[training_part:]

x_train, y_train = np.concatenate(
                        [x_train, np.flip(x_train, 2)]), \
                   np.concatenate([y_train, y_train])
num = x_train.shape[0]
# print('...', num, 'training samples, and',
#              x_val.shape[0], 'testing samples.')
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
input_shape = x_train[0].shape # (?, 48, 48, 1)

datagen = ImageDataGenerator(
    horizontal_flip=True,
    zoom_range=0.1,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1
    )
datagen.fit(x_train) #, augment=True)

model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))
model.add(Conv2D(64, (5, 5), padding='same'))
model.add(LeakyReLU(0.2))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) 
model.add(Dropout(0.4))

model.add(BatchNormalization(input_shape=input_shape))
model.add(Conv2D(128, (5, 5), padding='same'))
model.add(LeakyReLU(0.2))
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.4))

model.add(BatchNormalization(input_shape=input_shape))
model.add(Conv2D(256, (5, 5), padding='same'))
model.add(LeakyReLU(0.2))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) 
model.add(Dropout(0.4))

model.add(BatchNormalization(input_shape=input_shape))
model.add(Conv2D(256, (5, 5), padding='same'))
model.add(LeakyReLU(0.2))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) 
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(256, use_bias=True))
model.add(LeakyReLU(0.2))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))          

now_n = now_name()
model.compile(optimizer = Adam(lr=1e-3), # 'adam'
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
checkpointer = ModelCheckpoint(
    filepath=(wght_path + 
       '/model_best_{}.h5'.format(now_n)),
    monitor='val_acc', verbose=1, save_best_only=True)

early_stopping = EarlyStopping(
    monitor='val_loss', patience=50, verbose=1,
    restore_best_weights=False)

history = model.fit_generator(
    datagen.flow(x_train, y=y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train) // batch_size, 
    epochs=epochs,
    verbose=1,
    validation_data=(x_val, y_val),
    callbacks=[checkpointer, early_stopping])

model.save(wght_path + '/model_final_%s.h5'%now_n)
                # creates a HDF5 file Final_'model_done.h5'