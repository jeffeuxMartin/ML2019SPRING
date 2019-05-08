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

onegrp = lambda ii: np.expand_dims(ii, 0)

epsilon = 5.0
Lnrmlmt = 5 # 5.6750
baselin = 0.895

imgs  = np.array([image.img_to_array(i) for i in [ \
    image.load_img(i) for i in sorted(glob(folder_dest + '/*'))]])
imgsR = resnet50.preprocess_input( imgs.copy())

model = resnet50.ResNet50(weights='imagenet') 


truth = np.array(
    [305, 883, 243, 559, 438, 990, 949, 853, 609, 582, 915, 455, 619,
       961, 630, 741, 455, 707, 854, 922, 129, 537, 672, 476, 299,  99,
       476, 251, 520, 923, 760, 582, 525, 317, 464, 478, 667, 961, 865,
       324,  33, 922, 142, 312, 302, 582, 948, 360, 789, 440, 746, 764,
       949, 480, 792, 900, 733, 327, 441, 882, 920, 839, 955, 555, 519,
       510, 888, 990, 430, 396,  97,  78, 140, 362, 705, 659, 640, 967,
       489, 937, 991, 887, 603, 467, 498, 879, 807, 708, 967, 472, 287,
       853, 971, 805, 719, 854, 471, 890, 572, 883, 476, 581, 603, 967,
       311, 873, 582,  16, 672, 780, 489, 685, 366, 746, 599, 912, 950,
       614, 348, 353,  21,  84, 437, 946, 746, 646, 544, 469, 597,  81,
       734, 719,  51, 293, 897, 416, 544, 415, 814, 295, 829, 759, 971,
       306, 637, 471,  94, 984, 708, 863, 391, 383, 417, 442,  38, 858,
       716,  99, 546, 137, 980, 517, 322, 765, 632, 595, 754, 805, 873,
       475, 455, 442, 734, 879, 685, 521, 640, 663, 720, 759, 535, 582,
       607, 859, 532, 113, 695, 565, 554, 311,   8, 385, 570, 480, 324,
       897, 738, 814, 253, 751])
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
    
    diff = np.clip(inverse_trans(imgsR[q] + grad_sgn[0] * epsilon) - imgs[q], -Lnrmlmt, Lnrmlmt)
    
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