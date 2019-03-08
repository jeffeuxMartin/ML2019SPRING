from __future__ import print_function
# Standard Python Libraries
import csv
import os
import sys
import json
import time
import math
from argparse import ArgumentParser
### import glob

# Allowed Library
import numpy as np
### import scipy
### import pandas as pd

### """import numpy.linalg.lstsq # forbidden!"""

# hw1.py test predict -t train -lr lr -lb lambda -it iter -w weight -m mode
parser = ArgumentParser()
parser.add_argument("test", help="testing file")
parser.add_argument("predict", help="predict results file")
parser.add_argument("-t", "--train", help="training file", dest="train", default="train.csv")
parser.add_argument("-m", "--mode", help="mode if training or testing", dest="mode", default="test")
parser.add_argument("-it", help="iteration of training", dest="it", default="50000")
parser.add_argument("-lr", "--lr", help="learning rate", dest="lr", default="300")
parser.add_argument("-lb", "--lambda", help="lambda of regularization", dest="lambda_", default="0.1")
parser.add_argument("-w", "--weight", help="weight file", dest="weight", default="model.npy")
parser.add_argument("-nw", "--new_weight", help="new weight file", dest="newweight", default="")
args = parser.parse_args()

it, lr, lamb = int(args.it), eval(args.lr), eval(args.lambda_)

if os.path.isfile(args.weight):
    if args.weight.split('.')[-1] == 'npy':
        w = np.loadtxt(args.weight).reshape(-1, 1) 
    elif args.weight.split('.')[-1] == 'json':
        with open(args.weight, 'r', encoding='big5', newline='') as f:   
            w = np.array(json.load(f)).reshape(-1, 1) 
    

if os.path.isfile(args.train) and (args.mode == 'tr' or args.mode == 'train'):
    with open(args.train, 'r', encoding='big5', newline='') as f:   
        data = np.array([[eval(data) if data != 'NR' else 0 
           for data in r[3:]] for r in ([r for r in csv.reader(f)][1:])])
        data_c = data.reshape(12, 20, 18, 24).transpose(2, 0, 1, 3).reshape(18, 12, -1)

    # lowest = np.array([feat.min() for feat in data_c])
    # rangin = np.array([feat.max() - feat.min() for feat in data_c])
    # s_mean = np.array([feat.mean() for feat in data_c])
    # s_stdd = np.array([feat.std() for feat in data_c])

    # feat_scal_u, feat_scal_s = False, False
    # if feat_scal_u:
    #     data_c = np.array([(feat - lowest[n]) / rangin[n] for n, feat in enumerate(data_c)])
    # elif feat_scal_s:
    #     data_c = np.array([(feat - s_mean[n]) / s_stdd[n] for n, feat in enumerate(data_c)])

    pm25 = data_c[9]  # pm25 with 12 x 480
    data = data_c.reshape(18, 12, 20, 24).transpose(1, 2, 3, 0).reshape(12, 480, 18) 
    # data with 12 x 480 x 18
    
    feat_x, feat_y = [], []
    for month in data:  # month with 480 x 18
        for hour in range(480 - 10 + 1): # 0 ~ 469
            feat_x.append(month[hour:(hour + 10 - 1)])
            feat_y.append(month[hour + 10 - 1][9])
    feat_x, feat_y = np.array(feat_x).reshape(len(feat_x), -1), np.array(feat_y).reshape(-1, 1)
    dim_data, dim_w_b = feat_x.shape; dim_w_b += 1  # len = 5652
    X, Y = np.concatenate((feat_x, np.ones((len(feat_x), 1))), axis = 1), feat_y
     
    if not os.path.isfile(args.weight):
        w = np.zeros((dim_w_b, 1))

    mean_ = np.mean(X, axis = 0) 
    std_ = np.std(X, axis = 0)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if not std_[j] == 0 :
                X[i][j] = (X[i][j]- mean_[j]) / std_[j]

    prev_gra = np.zeros((18 * 9 + 1, 1)) # adagrad
    prev_loss = math.inf

    print("learning rate =", lr)
    lr_f = np.array([[lr]] * dim_w_b)
    for i in range(it):
        loss = (sum((Y - X.dot(w)) ** 2) / dim_data) ** 0.5
        if loss > prev_loss + 0.1:
            break
        elif loss - prev_loss > 0.0001:
            input("\nKeep going?")
        if loss > 1e20:
            break
        print("\riteration %14d / %14d : Loss = %.4f               " % (i + 1, it, loss), end='\r')
        grad = -2 * X.T.dot(Y - X.dot(w))
        prev_gra += grad ** 2
        ada = np.sqrt(prev_gra) + 0.005
        w -= lr_f * grad / ada
        if i % 500 == 0:
            if args.newweight == '':
                np.savetxt(args.weight, w, delimiter=',')
            else:
                np.savetxt(args.newweight, w, delimiter=',')

    print()
    if args.newweight == '':
        np.savetxt(args.weight, w, delimiter=',')
    else:
        np.savetxt(args.newweight, w, delimiter=',')

with open(args.test, 'r', encoding='big5', newline='') as f:   
    tdata = np.array([[eval(tdata) if tdata != 'NR' else 0 
       for tdata in r[2:]] for r in csv.reader(f)])
tdata_c = tdata.reshape(-1, 18, 9).transpose(1, 0, 2)
# if feat_scal_u:
#     tdata_c = np.array([(feat - lowest[n]) / rangin[n] for n, feat in enumerate(tdata_c)])
# elif feat_scal_s:
#     tdata_c = np.array([(feat - s_mean[n]) / s_stdd[n] for n, feat in enumerate(tdata_c)])
tpm25 = tdata_c[9]
tdata = tdata_c.transpose(1, 2, 0).reshape(-1, 9 * 18)

tX = np.concatenate((tdata, np.ones((len(tdata), 1))), axis = 1)
# for i in range(tX.shape[0]):
#     for j in range(tX.shape[1]):
#         if not std_[j] == 0 :
#             tX[i][j] = (tX[i][j]- mean_[j]) / std_[j]
res = tX.dot(w)

with open(args.predict, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
  
    writer.writerow(['id', 'value'])
    for ct3, row3 in enumerate(res):
        writer.writerow(['id_'+str(ct3), row3.item()])
