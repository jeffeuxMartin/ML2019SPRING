print()
import numpy as np
import data_reader
from argparse import ArgumentParser
import sys

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
x = data
label, y = data_reader.reader_res(args.dataY)
print(x)
print(y)
print('============================The program starts here...============================')
w, b = np.zeros((dim,)), np.zeros((1,))
iterations, lr = int(args.iterations), eval(args.lr)
sigmoid = lambda z: np.clip(1.0 / (1.0 + np.exp(-z)), 0.00000000000001, 0.99999999999999)


"""
for it in range(iterations):
    crsentr, w_grad, b_grad, res = 0.0, 0.0, 0.0, np.zeros((num,))
    for it in range(num):
        res[it] = sigmoid(np.dot(x[it], np.transpose(w)) + b)
        # crsentr += -(y[it]*np.log(fwb(x[it]))+(1-y[it])*np.log(1-fwb(x[it])))
        crsentr += -(np.dot(y[it], np.log(res[it])) + np.dot((1 - y[it]), np.log(1 - res[it])))
        # grad += -(y[it] - fwb(x[it]))*x[it]
        w_grad += -1 * x * (y[it] - res[it])
        b_grad += -1 * (y[it] - res[it])
    crsentr /= float(num); w_grad /= float(num); b_grad /= float(num)
    # res = sigmoid(data_b.dot(w_b))
    # clsed = np.array([1 if ans > 0.5 else 0 for ans in res]).astype(int)
    # acc = 1 - sum(abs(y-clsed)) / num
    # print("\rTry %7d: loss = %12.7lf, Score = %.5f "%(it + 1, crsentr / num, acc), end='')
    print("\rTry %7d: loss = %12.7lf"%(it + 1, crsentr), end='')
        # if acc > 0.84434:
        # print(' ==> Simple Baseline Passed??', end='')
    print('\r', end='')
    w -= lr * w_grad; b -= lr * b_grad
    w_b = np.concatenate((w, b))
    if it % 10 == 9:
    	with open('weights/model.npy', 'w') as fw:
    		np.savetxt(fw, w_b)

title_t, data_t = data_reader.reader_onehot('data/X_test')
num_t, dim_t = data_t.shape
data_bt = np.concatenate((data_t, np.ones(num_t).reshape(-1, 1)), 1) #.astype(int)

res_t = sigmoid(data_bt.dot(w_b))
classed = [1 if ans > 0.5 else 0 for ans in res_t]

with open('results/prediction_logi.csv', 'w') as fpr:
	fpr.write('id,label\n')
	for nw, dt in enumerate(classed):
		fpr.write(str(nw+1)+','+str(dt)+'\n')
print()
"""