import time, os, sys

import numpy as np; np.random.seed(0)
import pandas as pd

from tensorflow.python.keras import layers, backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing import image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

img = np.zeros((40000, 32, 32, 3))
for number in range(1, 40000 + 1):
    print("\rLoading #{:06d}.jpg...".format(number), end='\r')
    img[number - 1] = image.img_to_array(image.load_img(
        sys.argv[1] + '/%06d.jpg' % number))
img = img / 255.

latent_dim = 96 
PCA_dim = 49 
intermediates = 300, 125 
filters = 64, 48, 16
n_iter = 300
epsilon_std = 0.1
batch_size, epochs = 48, 500

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

input_img = layers.Input(shape=(32, 32, 3))
x1 = layers.Conv2D(3, (2, 2), activation='relu', padding='same')(input_img)
x2 = layers.Conv2D(filters[0], (2, 2), strides=(2, 2), activation='relu', padding='same')(x1)
x3 = layers.MaxPooling2D((2, 2), padding='same')(x2)
x4 = layers.Conv2D(filters[1], (3, 3), strides=(1, 1),activation='relu', padding='same')(x3)
x5 = layers.Conv2D(filters[2], (3, 3), strides=(1, 1),activation='relu', padding='same')(x4)
x6 = layers.MaxPooling2D((2, 2), padding='same')(x5)
x7 = layers.Flatten()(x6)
x8 = layers.Dense(intermediates[0], activation='relu')(x7)
hidden = layers.Dense(intermediates[1], activation='relu')(x8)

z_mean = layers.Dense(latent_dim)(hidden)
z_log_var = layers.Dense(latent_dim)(hidden)
z_enc = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

z_hid = layers.Dense(intermediates[1], activation='relu')(z_enc)
z8 = layers.Dense(intermediates[0], activation='relu')(z_hid)
z7 = layers.Dense(x7.get_shape()[-1].value, activation='relu')(z8)
z6 = layers.Reshape((x6.get_shape()[1].value, x6.get_shape()[2].value, filters[2]))(z7)
z5 = layers.UpSampling2D((2, 2))(z6)
z4 = layers.Conv2DTranspose(filters[1], (3, 3), strides=(1, 1), activation='relu', padding='same')(z5)
z3 = layers.Conv2DTranspose(filters[0], (3, 3), strides=(1, 1), activation='relu', padding='same')(z4)
z2 = layers.UpSampling2D((2, 2))(z3)
z1 = layers.Conv2DTranspose(3, (2, 2), strides=(2, 2), activation='relu', padding='same')(z2)
outputimg = layers.Conv2DTranspose(3, (2, 2), strides=1, activation='sigmoid', padding='same')(z1)


autoencoder = Model(input_img, outputimg)
autoencoder.load_weights(sys.argv[4])
def encoder(inp): 
    return K.function([autoencoder.layers[0].input], 
        [autoencoder.layers[12].output])([inp])[0]

latents = encoder(img)

# K.clear_session()

pca = PCA(n_components=PCA_dim,
	      copy=False,
	      whiten=True,
	      svd_solver='full'
	)
pca.fit(latents)

PCALat = pca.transform(latents)
print("original shape:   ", latents.shape)
print("transformed shape:", PCALat.shape)


kmeans = KMeans(n_clusters=2,
	random_state=0,
	max_iter=n_iter
	)
kmeans.fit(PCALat)
Origin = kmeans.labels_

print("original shape:   ", PCALat.shape)
print("transformed shape:", Origin.shape)

test_file = pd.read_csv(sys.argv[2])
test_cases = np.stack((test_file['image1_name'].to_numpy(), test_file['image2_name'].to_numpy())).T

compared = [1 if Origin[A - 1] == Origin[B - 1] else 0 for A, B in test_cases]
with open(sys.argv[3], 'w') as fw:
    fw.write('id,label\n')
    for _id, _label in enumerate(compared):
        fw.write('{},{}\n'.format(_id, _label))