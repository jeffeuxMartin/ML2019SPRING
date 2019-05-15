import sys, os, argparse, time
from glob import glob

import numpy as np
np.random.seed(0)

import pandas as pd

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers
from keras.preprocessing import image

from tensorflow.python.keras.layers \
    import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
# K.tensorflow_backend._get_available_gpus()

folder = './data/images'
split_rate = 0.1
# # total = len(glob(os.path.join(folder, '*')))
total = 40000
start_time = time.time()

# img = image.img_to_array(image.load_img(os.path.join(folder, '%06d.jpg' % 1)))
# img = np.zeros((total, *(img.shape)))
img = np.zeros((40000, 32, 32, 3))
if os.path.isfile('IM.npy'):
    img = np.load('IM.npy')
else:
    for number in range(1, total + 1):
        print("\rLoading #{:06d}.jpg...".format(number), end='\r')
        img[number - 1] = image.img_to_array(image.load_img(
            folder + '/%06d.jpg' % number))
    np.save('IM', arr=img)
print(time.time() - start_time, 'seconds...')
import time as t
now =t.strftime("%Y%m%d_%H%M%S",t.gmtime(t.time()+8*60*60))

val_margin = int(total * (1-split_rate))
img = img / 255.
x_train = img[:val_margin]
x_test = img[val_margin:]

im = img[0]





input_img = layers.Input(shape=im.shape)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.Conv2D(12, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(24, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(18, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(48, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(36, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

z = layers.Conv2D(36, (3, 3), activation='relu', padding='same')(encoded)
z = layers.Conv2D(48, (3, 3), activation='relu', padding='same')(encoded)
z = layers.UpSampling2D((2, 2))(z)
z = layers.Conv2D(18, (3, 3), activation='relu', padding='same')(z)
z = layers.Conv2D(24, (3, 3), activation='relu', padding='same')(z)
z = layers.UpSampling2D((2, 2))(z)
z = layers.Conv2D(12, (3, 3), activation='relu', padding='same')(z)
z = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(z)
z = layers.UpSampling2D((2, 2))(z)
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(z)

autoencoder = Model(input_img, decoded)
autoencoder.summary()





# input('qq')
encoder = Model(input_img, encoded)
layersLnumEn = len(encoder.layers)
encoded_input = layers.Input(shape=encoder.output_shape[1:])
r = autoencoder.layers[layersLnumEn](encoded_input)
for ly in autoencoder.layers[layersLnumEn + 1:-1]:
	r = ly(r)
re_decoded = autoencoder.layers[-1](r)
decoder = Model(encoded_input, re_decoded)

autoencoder.compile(
    # optimizer='adadelta', 
	optimizer='adam', 
    # loss='binary_crossentropy',
	loss='mse',
	metrics=['accuracy'],
	)

# input('check')
Colab = True
if Colab:
    from tensorflow.python.keras.callbacks import ModelCheckpoint
    from tensorflow.python.keras.callbacks import EarlyStopping
# 	AttributeError
else:
    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import EarlyStopping

	
checkpointer = ModelCheckpoint(
        # filepath=('model_best_{}.h5'.format(now)),
        filepath=(sys.argv[1]),
        # {epoch:02d}-{val_acc:.2f}.hdf5
        monitor='val_acc', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(
        monitor='val_loss', patience=30, verbose=1,
        restore_best_weights=False)

train_his = autoencoder.fit(
	x_train, x_train,
    epochs=50,
    # batch_size=32,
    batch_size=96,
    shuffle=True,
    validation_data=(x_test, x_test),
    callbacks=[\
        # TensorBoard(log_dir='/tmp/autoencoder')\
        checkpointer,
        early_stopping,
    ]
    )



"""input_img = Input(shape=im.shape)
x = layers.Conv2D(16, 2)(input_img)
x = layers.Conv2D(16, 2)(x)
x = layers.Conv2D(32, 2)(x)
x = layers.Conv2D(32, 2)(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)
x = layers.Conv2D(32, 2)(x)
x = layers.Conv2D(32, 2)(x)
x = layers.Conv2D(64, 2)(x)
x = layers.Conv2D(64, 2)(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)
x = layers.Conv2D(96, 2)(x)
x = layers.Conv2D(96, 2)(x)
x = layers.MaxPool2D(pool_size=(3, 3))(x)
x = layers.Flatten()(x)
x = layers.Dense(48)(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(24)(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(8)(x)
encoded = layers.Dropout(0.4)(x)

z = layers.Dense(24)(encoded)
z = layers.Dense(48)(z)
z = layers.Dense(96)(z)
z = layers.Reshape((1, 1, 96))(z)
z = layers.UpSampling2D((3, 3))(z)
z = layers.Conv2D(96, (2, 2), activation='relu', padding='same')(z)
z = layers.Conv2D(96, (2, 2), activation='relu', padding='same')(z)
z = layers.UpSampling2D((2, 2))(z)
z = layers.Conv2D(64, (2, 2), activation='relu', padding='same')(z)
z = layers.Conv2D(64, (2, 2), activation='relu', padding='same')(z)
z = layers.Conv2D(32, (2, 2), activation='relu', padding='same')(z)
z = layers.Conv2D(32, (2, 2), activation='relu', padding='same')(z)
z = layers.UpSampling2D((2, 2))(z)
z = layers.Conv2D(32, (2, 2), activation='relu', padding='same')(z)
z = layers.Conv2D(32, (2, 2), activation='relu', padding='same')(z)
z = layers.Conv2D(16, (2, 2), activation='relu', padding='same')(z)
z = layers.Conv2D(16, (2, 2), activation='relu', padding='same')(z)
decoded = layers.Conv2D(1, (2, 2), activation='sigmoid', padding='same')(z)

encoder = Model(input_img, encoded)
autoencoder = Model(input_img, decoded)
autoencoder.summary()"""


"""model = Sequential()
model.add(layers.Conv2D(16, 2, input_shape=im.shape))
model.add(layers.Conv2D(16, 2))
model.add(layers.Conv2D(32, 2))
model.add(layers.Conv2D(32, 2))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Conv2D(32, 2))
model.add(layers.Conv2D(32, 2))
model.add(layers.Conv2D(64, 2))
model.add(layers.Conv2D(64, 2))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Conv2D(96, 2))
model.add(layers.Conv2D(96, 2))
model.add(layers.MaxPool2D(pool_size=(3, 3)))
model.add(layers.Flatten())
model.add(layers.Dense(48))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(24))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(8))
model.add(layers.Dropout(0.4))
# model.summary()"""
