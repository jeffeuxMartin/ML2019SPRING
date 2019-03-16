print()
import numpy as np
import data_reader
from argparse import ArgumentParser
import sys
from numpy.linalg import inv, det, slogdet

parser = ArgumentParser()
parser.add_argument("dataX", help="Training one-hot data X.")
parser.add_argument("dataY", help="Training result data Y.")
parser.add_argument("-i", "--iter", "--iteration", help="Number of iterations.", 
                    dest="iterations", default="100")
parser.add_argument("-lr", "--lr", help="Learning rate.", 
                    dest="lr", default="8e-15")
"""Usage of argparse
parser.add_argument('--foo', help='foo help')
parser.add_argument('--text', '-t', type=str, required=True, help='Text for program')
parser.add_argument("-o", "--optional-arg", help="optional argument", dest="opt", default="default")
parser.print_help()
"""
args = parser.parse_args()
print(args)
title, data = data_reader.reader_onehot(args.dataX)
num, dim = data.shape

x = (data - data.mean(0)) / data.std(0)
MM = data.mean(0)
SS = data.std(0)
label, y = data_reader.reader_res(args.dataY)
print(x)
print(y)
print('============================The program starts here...============================')
w, b = np.zeros((dim,)), np.zeros((1,))
iterations, lr = int(args.iterations), eval(args.lr)
sigmoid = lambda z: np.clip(1.0 / (1.0 + np.exp(-z)), 0.00000000000001, 0.99999999999999)

w, b = np.zeros(dim), np.array([0])
# w_b = np.concatenate((w, b))
# w_b = np.random.random(dim + 1)

iterations, lr = int(args.iterations), eval(args.lr)
# sigmoid(x.dot(w_b))
# sigmoid(data.dot(w) + b)

grp0, grp1 = [], []
for n in range(num):
    if y[n] == 1:
        grp1.append(x[n])
    if y[n] == 0:
        grp0.append(x[n])
grp0, grp1 = np.array(grp0), np.array(grp1)
ct_0, ct_1 = len(grp0), len(grp1)
mean0, mean1 = grp0.mean(0), grp1.mean(0)

Var0 = Var1 = np.zeros((dim, dim))
Cov = lambda v: v.reshape(-1, 1) * v.reshape(1, -1)
for n in range(num):
    if y[n] == 1:
        Var1 += Cov(x[n] - mean1)
    if y[n] == 0:
        Var0 += Cov(x[n] - mean0)
Var0 /= ct_0; Var1 /= ct_1
Var = (float(ct_0) * Var0 + float(ct_1) * Var1) / num
antiVar = inv(Var)

pre = x.dot((mean0 - mean1).dot(antiVar))  \
     - 1/2 * mean0.dot(antiVar).dot(mean0) \
     + 1/2 * mean1.dot(antiVar).dot(mean1) \
     + np.log(ct_0 / ct_1)

prepre = sigmoid(
     x.dot((mean0 - mean1).dot(antiVar))  \
     - 1/2 * mean0.dot(antiVar).dot(mean0) \
     + 1/2 * mean1.dot(antiVar).dot(mean1) \
     + np.log(ct_0 / ct_1) )

preprepre = [1 if i > 0.5 else 0 for i in prepre]

# def predict()
print('\n==================================Testing starts!==================================')



titlet, datat = data_reader.reader_onehot('data/X_test')
numt, dimt = datat.shape
xt = (datat - MM) / SS
# w_b = np.concatenate((w, b))
# with open(args.weights, 'r') as f:
#     w_b = np.loadtxt(f)
# w, b = w_b[:-1], w_b[-1] 

# sigmoid = lambda z: np.clip(1.0 / (1.0 + np.exp(-z)), 
                            # 0.00000000000001, 0.99999999999999)
# res = sigmoid(data_bt.dot(w_b))
# res = sigmoid(np.dot(x, np.transpose(w)) + b)
# classed = [1 if ans > 0.5 else 0 for ans in res]
prepreq = sigmoid(
     xt.dot((mean0 - mean1).dot(antiVar))  \
     - 1/2 * mean0.dot(antiVar).dot(mean0) \
     + 1/2 * mean1.dot(antiVar).dot(mean1) \
     + np.log(ct_0 / ct_1) )

preprepreq = [1 if i <= 0.5 else 0 for i in prepreq]

with open('results/gen2.csv', 'w') as fpr:
	fpr.write('id,label\n')
	for nw, dt in enumerate(preprepreq):
		fpr.write(str(nw+1)+','+str(dt)+'\n')
print('\n==============================Prediction finished!=================================')