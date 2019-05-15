import sys
from tensorflow.keras.models import load_model 
from tensorflow.keras import backend as K
md = load_model("model_190515_045717.h5")   

en = K.function([md.layers[0].input], [md.layers[8].output])

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

folder = './data/images'
split_rate = 0.1
# # total = len(glob(os.path.join(folder, '*')))
total = 40000
start_time = time.time()

# # img = image.img_to_array(image.load_img(os.path.join(folder, '%06d.jpg' % 1)))
# # img = np.zeros((total, *(img.shape)))
# img = np.zeros((40000, 32, 32, 3))
# # for number in range(1, total + 1):
# #     print("\rLoading #{:06d}.jpg...".format(number), end='\r')
# #     img[number - 1] = image.img_to_array(image.load_img(
# #         folder + '/%06d.jpg' % number))
img = np.load('IM.npy')
print(time.time() - start_time, 'seconds...')

# val_margin = int(total * (1-split_rate))
# x_train = img[:val_margin] / 255.
# x_test = img[val_margin:] / 255.
img = img / 255.

# latent = en([x_test])[0].reshape(-1, 2 * 2 * 4)
latent = en([img])[0].reshape(-1, 2 * 2 * 4)
# latent = np.load('lat.npy')
comp = pd.read_csv('./data/test_case.csv').values.T[1:].T

res = []
from numpy.linalg import norm
for n, (a, b) in enumerate(comp):
    print('\rDoing {}...'.format(n), end='\r')
    u, v = latent[a-1], latent[b-1]
    cos_sim = np.dot(u, v)/(norm(u)*norm(v))
    res.append(int(np.round(cos_sim)))
print()

with open("pred_190515_045717.csv", 'w') as f:
    f.write('id,label\n')
    for n in range(len(res)):
        f.write('{},{}\n'.format(n, res[n]))

