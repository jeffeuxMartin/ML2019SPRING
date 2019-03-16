import keras
from keras.models import Sequential
from keras.layers import Dense
import data_reader as dr
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from keras.utils import np_utils

np.random.seed(10)

title, data = dr.reader_onehot('../data/X_train')
num, dim = data.shape
x = (data - data.mean(0)) / data.std(0)
x_tr, x_te = x[:num//5], x[num//5:]
title, y = dr.reader_res('../data/Y_train')
y = np_utils.to_categorical(y)
y_tr, y_te = y[:num//5], y[num//5:]

model = Sequential()
model.add(Dense(100, input_dim=dim, activation='relu'))
model.add(Dense(units=2, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam',
	          metrics=['accuracy'])
# for step in range(1 + 300):
#     cost = model.train_on_batch(data, y)
#     print('Iteration {}: loss = {}'.format(step, cost))
his = model.fit(x_tr, y_tr, epochs=20, batch_size=200)#, verbose=2)

scores = model.evaluate(x_te, y_te)
print()
print('Accuracy=', scores[1])
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))