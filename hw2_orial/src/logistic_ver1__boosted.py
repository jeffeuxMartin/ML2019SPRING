# print()
import numpy as np
import pandas as pd
# import data_reader as dt
from argparse import ArgumentParser
import sys
from numpy.linalg import inv, det, slogdet

parser = ArgumentParser()
parser.add_argument("dataX", help="Training one-hot data X.")
parser.add_argument("dataY", help="Training result data Y.")
parser.add_argument("-i", "-it", "--iter", "--iteration", help="Number of iterations.", 
                    dest="iterations", default="2000", type=int)
parser.add_argument("-lr", "--lr", help="Learning rate.", 
                    dest="lr", default="1e-4")
parser.add_argument("-f", "--feat", help="Feature scaling used.", 
                    dest="feat", default="weights/feat_scale.npy")
parser.add_argument("weights", help="The weights of the model.")
args = parser.parse_args()

data = np.array(pd.read_csv(args.dataX))
num, dim = data.shape
mean_, std_ = data.mean(0), data.std(0)
np.savetxt(args.feat, np.array([mean_, std_]))

x_ori = data.copy()
data = (data - mean_) / std_
data_b = np.concatenate((data, np.ones(num).reshape(-1, 1)), 1) #.astype(int)
x = data_b

y = np.array(pd.read_csv(args.dataY)).reshape(-1) 
# print('==============================Finish data loading...==============================')
w, b = np.zeros((dim,)), np.zeros((1,))
w_b = np.concatenate((w, b))
iterations, lr = int(args.iterations), eval(args.lr)
sigmoid = lambda z: np.clip(1.0 / (1.0 + np.exp(-z)), 
                            0.00000000000001, 0.99999999999999)
res = sigmoid(data_b.dot(w_b))

for it in range(iterations):
    loss, grad = 0.0, 0.0
    # for d in range(num):
    #     loss += -(y[d]*np.log(sigmoid(x[d].dot(w_b)))+(1-y[d])*np.log(1-sigmoid(x[d].dot(w_b))))
    #     grad += -(y[d] - sigmoid(x[d].dot(w_b)))*x[d]
    loss = -sum(y*(np.log(sigmoid(x.dot(w_b)))) + (1-y)*(np.log(1-sigmoid(x.dot(w_b)))))
    grad = -(y-sigmoid(x.dot(w_b))).dot(x)

    # res = sigmoid(data_b.dot(w_b))
    # clsed = np.array([1 if ans > 0.5 else 0 for ans in res]).astype(int)
    # acc = 1 - sum(abs(y-clsed)) / num
    # print("\rIteration %7d / %7d : loss = %12.7lf, Score = %.5f "%(it + 1, args.iterations, loss / num, acc), end='')    
    print("\rIteration %7d / %7d : loss = %12.7lf"%(it + 1, args.iterations, loss / num), end='')    
    # if acc > 0.85565:
    #     print(' ==> Strong Baseline Passed??', end='')
    # elif acc > 0.84434:
    #     print(' ==> Simple Baseline Passed??', end='')
    print('\r', end='')
    w_b -= lr * grad
    if it % 10 == 9:
    	with open(args.weights, 'w') as fw:
    		np.savetxt(fw, w_b)
print('\n')

with open(args.weights, 'w') as fw:
    np.savetxt(fw, w_b)