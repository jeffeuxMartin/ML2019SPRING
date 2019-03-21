print()
import numpy as np
import pandas as pd
# import data_reader as dt
from argparse import ArgumentParser
import sys
from numpy.linalg import inv, det, slogdet

parser = ArgumentParser()
parser.add_argument("dataX", help="Training one-hot data X.")
parser.add_argument("dataY", help="Training result data Y.")
parser.add_argument("-i", "--iter", "--iteration", help="Number of iterations.", 
                    dest="iterations", default="120", type=int)
parser.add_argument("-lr", "--lr", help="Learning rate.", 
                    dest="lr", default="5e-5")
parser.add_argument("-f", "--feat", help="Feature scaling used.", 
                    dest="feat", default="weights/feat_norma.npy")
parser.add_argument("weights", help="The weights of the model.")
args = parser.parse_args()
# print('============================The program starts here...============================')
# title, data = dt.reader_onehot(args.dataX)
data = np.array(pd.read_csv(args.dataX))
num, dim = data.shape
mean_, std_ = data.min(0), data.max(0) - data.min(0)
np.savetxt(args.feat, np.array([mean_, std_]))

x_ori = data.copy()
data = (data - mean_) / std_
x = data

# label, y = dt.reader_res(args.dataY)
y = np.array(pd.read_csv(args.dataY)).reshape(-1) 
# print('==============================Finish data loading...==============================')

sigmoid = lambda z: np.clip(1.0 / (1.0 + np.exp(-z)), 
                            0.00000000000001, 0.99999999999999)
grpdata_o = ([], [])
for n in range(num):
    grpdata_o[y[n]].append(x[n])
grpdata = (np.array(grpdata_o[0]), np.array(grpdata_o[1]))

ct = (len(grpdata[0]), len(grpdata[1]))
mean_d = (grpdata[0].mean(0), grpdata[1].mean(0))
Var = [np.zeros((dim, dim)), np.zeros((dim, dim))]
for k in [0, 1]:
    for v in grpdata[k]:
        gr = v - mean_d[k]
        Var[k] += gr.reshape(-1, 1).dot(gr.reshape(1,-1))
    Var[k] /= ct[k]
tVar = (float(ct[0]) * Var[0] + float(ct[1]) * Var[1]) / num
antiVar = inv(tVar)

w = (mean_d[1] - mean_d[0]).dot(antiVar)
b = - 1/2 * mean_d[1].dot(antiVar).dot(mean_d[1]) \
    + 1/2 * mean_d[0].dot(antiVar).dot(mean_d[0]) \
    + np.log(ct[1] / ct[0])
w_b = np.concatenate((w.reshape(-1, 1), np.array(b).reshape(1, 1)), 0)

with open(args.weights, 'w') as fw:
    np.savetxt(fw, w_b)