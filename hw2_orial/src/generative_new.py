import numpy as np
import pandas as pd
from argparse import ArgumentParser
import sys
from numpy.linalg import inv, det, slogdet

#run src/generative_new.py data/X_train data/Y_train ~/Desktop/model1.npy
parser = ArgumentParser()
parser.add_argument("dataX", help="Training one-hot data X.")
parser.add_argument("dataY", help="Training result data Y.")
parser.add_argument("-i", "--iter", "--iteration", help="Number of iterations.", 
                    dest="iterations", default="120", type=int)
parser.add_argument("-lr", "--lr", help="Learning rate.", 
                    dest="lr", default="5e-5")
parser.add_argument("-f", "--feat", help="Feature scaling used.", 
                    dest="feat", default="weights/feat_scale.npy")
parser.add_argument("weights", help="The weights of the model.")
args = parser.parse_args()

sigmoid = lambda z: np.clip(1.0 / (1.0 + np.exp(-z)), 
                            0.00000000000001, 0.99999999999999)

data = np.array(pd.read_csv(args.dataX))
num, dim = data.shape

all_mean, all_std = data.mean(0), data.std(0)
scaled_data_A = (data - all_mean) / all_std
np.savetxt(args.feat, np.array([all_mean, all_std]))

cdt1, cdt2 = [data.T[0:2].T, data.T[3:6].T]
muc1, muc2 = cdt1.mean(0), cdt2.mean(0)
sgc1, sgc2 = cdt1.std(0), cdt2.std(0)
new_mean = np.concatenate([muc1, np.zeros(1), muc2, np.zeros(100)], 0)
new_std = np.concatenate([sgc1, np.ones(1), sgc2, np.ones(100)], 0)
np.savetxt(args.feat, np.array([new_mean, new_std]))

scaled_data_B = (data - new_mean) / new_std
ddt0 = data.T[2:3].T
ddt = [data.T[6:15].T, data.T[15:31].T, data.T[31:38].T, 
       data.T[38:53].T, data.T[53:59].T, data.T[59:].T]

x = scaled_data_B.copy()
y = np.array(pd.read_csv(args.dataY)).reshape(-1)

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