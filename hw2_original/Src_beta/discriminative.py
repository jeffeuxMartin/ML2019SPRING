print()
import numpy as np
import src.data_reader as dt
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
args = parser.parse_args()
print('============================The program starts here...============================')
title, data = dt.reader_onehot(args.dataX)
num, dim = data.shape
mean_, std_ = data.mean(0), data.std(0)
x_ori = data.copy()
data = (data - mean_) / std_
data_b = np.concatenate((data, np.ones(num).reshape(-1, 1)), 1) #.astype(int)
x = data_b

label, y = dt.reader_res(args.dataY)
print('==============================Finish data loading...==============================')
w, b = np.zeros((dim,)), np.zeros((1,))
w_b = np.concatenate((w, b))
iterations, lr = int(args.iterations), eval(args.lr)
sigmoid = lambda z: np.clip(1.0 / (1.0 + np.exp(-z)), 
                            0.00000000000001, 0.99999999999999)
fwb = lambda xxx: sigmoid(xxx.dot(w_b))
res = sigmoid(data_b.dot(w_b))

for it in range(iterations):
    loss, grad = 0.0, 0.0
    for dd in range(num):
        loss += -(y[dd]*np.log(fwb(x[dd]))+(1-y[dd])*np.log(1-fwb(x[dd])))
        grad += -(y[dd] - fwb(x[dd]))*x[dd]
    res = sigmoid(data_b.dot(w_b))
    clsed = np.array([1 if ans > 0.5 else 0 for ans in res]).astype(int)
    acc = 1 - sum(abs(y-clsed)) / num
    print("\rTry %7d: loss = %12.7lf, Score = %.5f "%(it + 1, loss / num, acc), end='')    
    if acc > 0.85565:
        print(' ==> Strong Baseline Passed??', end='')
    elif acc > 0.84434:
        print(' ==> Simple Baseline Passed??', end='')
    print('\r', end='')
    w_b -= lr * grad
    if it % 10 == 9:
    	with open(sys.argv[3], 'w') as fw:
    		np.savetxt(fw, w_b)

title_t, data_t = dt.reader_onehot('data/X_test')
num_t, dim_t = data_t.shape
x_t_ort = data_t.copy()
data_t = (data_t - mean_) / std_
data_bt = np.concatenate((data_t, np.ones(num_t).reshape(-1, 1)), 1) #.astype(int)

res_t = sigmoid(data_bt.dot(w_b))
classed = [1 if ans > 0.5 else 0 for ans in res_t]

with open(sys.argv[4], 'w') as fpr:
	fpr.write('id,label\n')
	for nw, dt in enumerate(classed):
		fpr.write(str(nw+1)+','+str(dt)+'\n')
print()
