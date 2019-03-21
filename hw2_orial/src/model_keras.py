import keras
from keras.models import Sequential
from keras.layers import Dense
import data_reader as dr
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras import optimizers
import sys

#np.random.seed(10)

title, data = dr.reader_onehot('../data/X_train')
num, dim = data.shape
x = (data - data.mean(0)) / data.std(0)
x_tr, x_te = x[:num//20], x[num//20:]
title, y = dr.reader_res('../data/Y_train')
# y = np_utils.to_categorical(y)
y_tr, y_te = y[:num//20], y[num//20:]

model = Sequential()
model.add(Dense(1, input_dim=dim, activation='sigmoid', use_bias=True))

#model.add(Dense(20, input_dim=dim, activation='relu', use_bias=True))
#model.add(Dense(1, activation='sigmoid', use_bias=True))

model.summary()

adam = optimizers.Adam(lr=float(sys.argv[3]))
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
# for step in range(1 + 300):
#     cost = model.train_on_batch(data, y)
#     print('Iteration {}: loss = {}'.format(step, cost))

early_stopping = EarlyStopping(monitor='val_loss', patience=200, verbose=2)
his = model.fit(x_tr, y_tr, validation_split=0.05, epochs=int(sys.argv[1]), batch_size=int(sys.argv[2]), verbose=1, callbacks=[early_stopping])

scores = model.evaluate(x_te, y_te)
print()
print('Accuracy=', scores[1])
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*10))

title, testx = dr.reader_onehot('../data/X_test')
prediction = model.predict((testx-data.mean(0))/data.std(0))
classed = [1 if ans > 0.5 else 0 for ans in prediction]

with open('../results/pred_keras.csv', 'w') as fpr:
    print("Predicted!")
    fpr.write('id,label\n')
    for nw, dt in enumerate(classed):
        fpr.write(str(nw+1)+','+str(dt)+'\n')
