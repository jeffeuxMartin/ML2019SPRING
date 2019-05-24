# bash cluster.sh <images path> <test_case.csv path> <prediction file path>
# e.g. bash cluster.sh  images/   test_case.csv   ans.csv

# python3 reproduce.py $1 $2 $3

# model_name=`date +"%y%m%d_%H%M%S"`
# python3 first_try.py "model_${model_name}.h5"
# python3 repr.py "model_${model_name}.h5" "pred_${model_name}.csv"



# model_name = !echo `TZ=":Asia/Taipei" date +"%y%m%d_%H%M%S"`
# model_name = model_name[0]
import sys
mysysargv = sys.argv
model_name = "190521_065402"
# myJeffname = 'predictions/prediction_{}.csv'.format(model_name)
myJeffname = mysysargv[3]

import time, os
import numpy as np
np.random.seed(0)
from tensorflow.python.keras import layers, backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing import image
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cluster import KMeans


weight_name = mysysargv[4]
folder = mysysargv[1]
reload_img = False

total = 40000
start_time = time.time()
img = np.zeros((40000, 32, 32, 3))
# if os.path.isfile('IM.npy') and not reload_img:
#     img = np.load('IM.npy')
# if os.path.isfile(mysysargv[1]) and not reload_img:
    # img = np.load(mysysargv[1])
# else:
for number in range(1, total + 1):
    print("\rLoading #{:06d}.jpg...".format(number), end='\r')
    img[number - 1] = image.img_to_array(image.load_img(
        folder + '/%06d.jpg' % number))
    # np.save('IM_new', arr=img)
# print(time.time() - start_time, 'seconds...')
img = img / 255.
im = img[0]

latent_dim = 96 # 108 
PCA_dim = 49 # 64

intermediates = 300, 125 # 250, 125 
filters = 64, 48, 16

n_iter = 300

epsilon_std = 0.1

batch_size, epochs = 48, 500

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

input_img = layers.Input(shape=im.shape)
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
autoencoder.load_weights(weight_name)
def encoder(inp): 
    return K.function([autoencoder.layers[0].input], [autoencoder.layers[12].output])([inp])[0]

latents = encoder(img)

K.clear_session()



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


# test_file = pd.read_csv('data/test_case.csv')
test_file = pd.read_csv(mysysargv[2])
test_cases = np.stack((test_file['image1_name'].to_numpy(), test_file['image2_name'].to_numpy())).T

compared = [1 if Origin[A - 1] == Origin[B - 1] else 0 for A, B in test_cases]
with open(myJeffname, 'w') as fw:
    fw.write('id,label\n')
    for _id, _label in enumerate(compared):
        fw.write('{},{}\n'.format(_id, _label))