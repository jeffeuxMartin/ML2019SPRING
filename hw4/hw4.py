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