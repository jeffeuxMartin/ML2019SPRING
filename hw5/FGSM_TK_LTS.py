from glob import glob
import sys

import time
stime = time.time()
import numpy as np
import pandas as pd
from PIL.Image import fromarray

import tensorflow as tf

from tensorflow.keras.applications import vgg16, resnet50
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
# K.tensorflow_backend._get_available_gpus()

session_counter = 0
start_num = 0

"""圖片預處理方式有三種caffe、tf、torch:
caffe : VGG16、VGG19、ResNet50
tf : Xception、InceptionV3、InceptionResNetV2、MobileNet、NASNet、MobileNetV2
torch : DenseNet
mode = caffe 
(will convert the images from RGB to BGR, then will zero-center each color\
 channel with respect to the ImageNet dataset)
減去ImageNet平均 BGR [103.939, 116.779, 123.68]
mode = tf 
( will scale pixels between -1 and 1 )
除以127.5，然後減 1。
mode = torch 
( will scale pixels between 0 and 1 and then will normalize each channel \
with respect to the ImageNet dataset)
除以255，減去ImageNet平均[0.485, 0.456, 0.406] ，除以標準差[0.229, 0.224, 0.225]。\
"""
folder_dest = sys.argv[1]
result_dest = sys.argv[2]
# folder_dest = "."
print('Hi:', folder_dest, '\n', result_dest)

def invTrsCf(t):
    return np.round(np.clip((np.array(
        t.transpose(2, 0, 1)[::-1]).transpose(1, 2, 0) 
            + [123.68, 116.779, 103.939]), 0, 255)).astype(np.int32)
def invTrsTf(t):
    return np.round(np.clip((t + 1) * 127.5, 0, 255)).astype(np.int32)
def invTrsTr(t):
    return np.round(np.clip(((t * [0.229, 0.224, 0.225])
           + [0.485, 0.456, 0.406]) * 255.0, 0, 255)).astype(np.int32)

inverse_trans = invTrsCf
def invBatch(btc): return np.array([inverse_trans(i) for i in btc]) 
def l_infty_(x, x_adv):
    return max((abs(x - x_adv).reshape(-1)) * 255.0)
def accuracy(ans_0, final):
    return len((ans_0 - final).nonzero()[0]) / 200.0
onegrp = lambda ii: np.expand_dims(ii, 0)

epsilon = 5.0
Lnrmlmt = 5 # 5.6750
baselin = 0.895

imgs  = np.array([image.img_to_array(i) for i in [ \
    image.load_img(i) for i in sorted(glob(folder_dest + '/*'))]])
imgsR = resnet50.preprocess_input( imgs.copy())

model = resnet50.ResNet50(weights='imagenet') 


truth = np.array(pd.read_csv('./labels.csv')['TrueLabel'])
xnois = np.zeros_like(imgsR)

onegrp = lambda ii: np.expand_dims(ii, 0)
inverse_trans = invTrsCf

def attacker(q):
    print('%3d'%q, end='\t')
    Grad = K.gradients(
        K.categorical_crossentropy(
            K.one_hot(truth, 1000)[q], model.output),
        model.input)
    Grad_sgn = K.sign(Grad)
    grad_sgn = sess.run(Grad_sgn, 
        feed_dict={
            model.input: onegrp(imgsR[q])})[0]  #dangerous
    done = imgsR[q] + grad_sgn[0] * epsilon
    okImg = inverse_trans(done)
    diff = okImg - imgs[q]
    diff = np.clip(diff, -Lnrmlmt, Lnrmlmt)
    
    fromarray(np.clip(imgs[q] + diff, 0, 255
        ).astype(np.uint8)).save(result_dest + '/' + '%03d'%q + '.png')

sess  = K.get_session() 
for qq in range(start_num, 200):
    attacker(qq)
    print('%8.3lf'%((time.time()-stime)/(qq - start_num + 1)*200/60), 'minutes?')
    session_counter += 1
    if session_counter > 25:
        K.clear_session(); session_counter = 0
        sess  = K.get_session() 
        model = resnet50.ResNet50(weights='imagenet') 

# %cd RR_hires50
# !tar -zcf ../RR_hires50.tgz *.png
# input()