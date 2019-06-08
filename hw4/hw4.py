from matplotlib import use
use('TkAgg')
# %matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
# import seaborn as sns
import sys
from PIL import Image
def show(pic): plt.imshow((1 - pic).reshape(48, 48))
from keras.models import load_model
from keras import backend as K
K.set_learning_phase(1)
import tensorflow as tf
np.random.seed(0)

model = load_model('model_best_0410_034513.h5')
layer_dict = dict([(layer.name, layer) for layer in model.layers])

class Gradienting(object):
    def __init__(self, model, output_index=0):
        input_tensors = [model.input]
        gradients = model.optimizer.get_gradients(
            model.output[0][output_index], model.input)
        self.compute_grads = K.function(
            inputs = input_tensors, outputs = gradients)
    def get_mask(self, input_image):
        x_v = np.expand_dims(input_image, axis=0)
        gradients = self.compute_grads([x_v])[0][0]
        return gradients
# print(sys.argv)
# input()
train = pd.read_csv(sys.argv[1])
x = np.array([np.array(pic.split()).astype(int) for pic in train['feature']]).reshape(-1, 48, 48, 1)
y = np.array(train['label'])

# angry disgust fear happy sad surprise neutral

num, task, target = len(x), dict(zip(range(7), [0]*7)), {}
for k in range(num):
    Yq, Yr = model.predict_classes(np.array(x[k]).reshape(1, 48, 48, 1)).item(), y[k]
    Good = Yq == Yr
#     print((k, Yq, Yr, Good))
    if Good:
        task[Yq] += 1
        if target.get(Yr) == None:
            target[Yr] = [k]
        else:
            target[Yr] = target[Yr] + [k]
    if min(list(task.values())) == 7:
        break
# target == {0: 10, 1: 533, 2: 2, 3: 7, 4: 58, 5: 15, 6: 4} 2630541
fine_pic = []

for i in range(7):
    chance = len(x[target[i]])
    for j in range(chance):
        grad_sal = Gradienting(model, y[target[i][j]]).get_mask(x[target[i][j]])
        # print(sum(grad_sal.reshape(-1)), '@', j, '/', chance)
        picture = plt.imshow(abs(grad_sal).reshape(48, 48), cmap = 'jet')
        plt.savefig(sys.argv[2] + '/fig1_{}.jpg'.format(i))
        # plt.imshow((1 - x[target[i][j]]).reshape(48, 48))
        # plt.savefig('ori1_{}.jpg'.format(i))
        if sum(grad_sal.reshape(-1)) != 0.0: 
            # print('\nresult =', j, target[i][j])
            fine_pic.append(target[i][j])
            break

from lime import lime_image
from skimage.segmentation import slic
from skimage.io import imread

def predict(input_):
    tranned = np.expand_dims(input_.transpose(3, 0, 1, 2)[0], axis=3)
    return model.predict(tranned)

x_train_rgb = x.repeat(3, axis=3)
explainer = lime_image.LimeImageExplainer()

def segmentation(input_):
    return slic(input_)
for nindx, idx in enumerate(fine_pic):
    explainer = lime_image.LimeImageExplainer()
    # Get the explaination of an image
    explaination = explainer.explain_instance(
#                             image=x[idx], 
                    image=x_train_rgb[idx], 
                    classifier_fn=predict,
                    segmentation_fn=segmentation
                )

    # # Get processed image
    image, mask = explaination.get_image_and_mask(
                    label=y[idx],
                    positive_only=False,
                    hide_rest=False,
                    num_features=5,
                    min_weight=0.0
                    )
    plt.axis('off')
    plt.imshow(image.astype('uint8'))
    plt.savefig(sys.argv[2] + '/fig3_{}.jpg'.format(nindx))
#     print('fig3_{}.jpg'.format(nindx))

import time
s_time = time.time()
X_train = x
# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    #x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def vis_img_in_filter(img, 
                      layer_name):
    img = np.array(img).reshape(-1, 48, 48, 1).astype(np.float64)[0].reshape(1, 48, 48, 1)
#     print(img.shape)
    layer_output = layer_dict[layer_name].output
    img_ascs = list()
    for filter_index in range(layer_output.shape[3]):
        # build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        loss = K.mean(layer_output[:, :, :, filter_index])

        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, model.input)[0]

        # normalization trick: we normalize the gradient
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([model.input], [loss, grads])

        # step size for gradient ascent
        step = 5.

#         img_asc = (np.random.random(img.shape) * 255.0).astype(int).astype(float)
        img_asc = (np.zeros_like(img)).astype(np.int32).astype(np.float64)
        
        for i in range(20):
            loss_value, grads_value = iterate([img_asc])
            img_asc += grads_value * step

        img_asc = img_asc[0]        
        img_ascs.append(deprocess_image(img_asc).reshape((48, 48)))

    plot_x, plot_y = 8, 8
    fig, ax = plt.subplots(plot_x, plot_y, figsize = (12, 12))
    fig.suptitle('Input image and %s filters' % (layer_name,))
    fig.tight_layout(rect = [0, 0, 1, 1], pad=0)
    for (x, y) in [(i, j) for i in range(plot_x) for j in range(plot_y)]:
        ax[x, y].imshow(img_ascs[x * plot_y + y], cmap = 'gray')#, interpolation="nearest")
        ax[x, y].axis('off')
    fig.savefig('fig2_1.jpg')
    return img_ascs
    
def vis_oup_in_filter(model, img, layer_name):
    img = img.reshape(-1, 48, 48, 1)[0].reshape(1, 48, 48, 1)
    get_outputs = K.function([model.layers[0].input],
                            [layer_dict[layer_name].output])
    kernels_output = get_outputs([img])[0]
    kernels_output = kernels_output.transpose(3, 0, 1, 2)
    channel, img_x, img_y, batch_sz= kernels_output.shape
    plot_x, plot_y = channel // 8, 8
    fig, ax = plt.subplots(plot_x, plot_y, figsize = (12, 12))
    fig.suptitle('Output images of %s filters' % (layer_name,))
    fig.tight_layout(rect = [0, 0, 1, 1], pad=0)
    for (x, y) in [(i, j) for i in range(plot_x) for j in range(plot_y)]:
        ax[x, y].imshow(kernels_output[x * plot_y + y].reshape(img_x, img_y), cmap = 'gray')
        ax[x, y].axis('off')
    fig.savefig('fig2_2.jpg')
    return kernels_output

img_ascs = vis_img_in_filter(x[fine_pic[1]], 'conv2d_1')
# print(time.time() - s_time)
kernels_output = vis_oup_in_filter(model, x[fine_pic[1]], 'conv2d_1')
# print(time.time() - s_time)