import keras
from keras.models import Sequential
from keras.layers import Dense
import data_reader as dr
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.callbacks import EarlyStopping


np.random.seed(10)

title, data = dr.reader_onehot('../data/X_train')
num, dim = data.shape
x = (data - data.mean(0)) / data.std(0)
x_tr, x_te = x[:num//10], x[num//10:]
title, y = dr.reader_res('../data/Y_train')
# y = np_utils.to_categorical(y)
y_tr, y_te = y[:num//10], y[num//10:]

model = Sequential()
model.add(Dense(1, input_dim=dim, kernel_initializer='normal', activation='sigmoid', use_bias=True))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# for step in range(1 + 300):
#     cost = model.train_on_batch(data, y)
#     print('Iteration {}: loss = {}'.format(step, cost))

early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
his = model.fit(x_tr, y_tr, validation_split=0.2, epochs=100, batch_size=64, verbose=1, callbacks=[early_stopping])

scores = model.evaluate(x_te, y_te)
print()
print('Accuracy=', scores[1])
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*10))

title, testx = dr.reader_onehot('../data/X_test')
prediction = model.predict((testx-data.mean(0))/data.std(0))
classed = [1 if ans > 0.5 else 0 for ans in prediction]

with open('../results/pred_keras.csv', 'w') as fpr:
    fpr.write('id,label\n')
    for nw, dt in enumerate(classed):
        fpr.write(str(nw+1)+','+str(dt)+'\n')
