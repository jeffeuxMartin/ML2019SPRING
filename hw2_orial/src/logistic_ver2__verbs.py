import numpy as np
import pandas as pd
from argparse import ArgumentParser
import sys
from numpy.linalg import inv, det, slogdet
import time
import shutil

parser = ArgumentParser()
parser.add_argument("dataX", help="Training one-hot data X.")
parser.add_argument("dataY", help="Training result data Y.")
parser.add_argument("-i", "-it", "--iter", "--iteration", help="Number of iterations.", 
                    dest="iterations", default="2000", type=int)
parser.add_argument("-lr", "--lr", help="Learning rate.", 
                    dest="lr", default="8e-6")
parser.add_argument("-f", "--feat", help="Feature scaling used.", 
                    dest="feat", default="weights/feat_scale.npy")
parser.add_argument("weights", help="The weights of the model.")
parser.add_argument("-rgl", "--lmbda", help="Regularization term.", 
                    dest="rgl", default="12")
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
w, b = np.zeros((dim,)), np.zeros((1,))
w_b = np.concatenate((w, b))
iterations, lr = int(args.iterations), eval(args.lr)
lmbda = eval(args.rgl)
sigmoid = lambda z: np.clip(1.0 / (1.0 + np.exp(-z)), 
                            0.00000000000001, 0.99999999999999)
res = sigmoid(data_b.dot(w_b))

for it in range(iterations):
    icon = shutil.get_terminal_size().columns - (10+7 + 3+7 + 8+(2+5) + 8+(1+12) + 10+(2+5)+ 2) - 29
    try:
        loss = -sum(y*(np.log(sigmoid(x.dot(w_b)))) + (1-y)*(np.log(1-sigmoid(x.dot(w_b))))) + lmbda * sum(w_b[:-1]**2)
        grad = -(y-sigmoid(x.dot(w_b))).dot(x) + 2 * lmbda * np.concatenate((w_b[:-1], np.array([0])), 0)

        res = sigmoid(data_b.dot(w_b))
        clsed = np.array([1 if ans > 0.5 else 0 for ans in res]).astype(int)
        acc = 1 - sum(abs(y-clsed)) / num
        print("\rIteration {:7d} / {:7d} : lr = {:.5f} loss = {:12.7f}, Score = {:.5f}  ".format(
            it + 1, iterations, lr, loss / num, acc), end='')
        if it == iterations - 1:
            it += (icon - 1) - (it % icon)
        print("[" + (it % icon - 1)*'=', end='')
        if (it % icon != 0) and (it % icon != (icon - 1)):
            print('>', end='')
        print((icon-(it % icon - 1) - 3)*'.'+']', end='')
        if acc > 0.85565:
            print(' ==> Strong Baseline Passed??', end='')
        elif acc > 0.84434:
            print(' ==> Simple Baseline Passed??', end='')
        print('\r', end='')
        w_b -= lr * grad
        if it % icon == (icon - 1):
            print()
            with open(args.weights, 'w') as fw:
                np.savetxt(fw, w_b)
        # time.sleep(0.1)
        # input()
        # if it % icon == (icon - 1):
            # print()
    except KeyboardInterrupt:
        break
print('\n')

with open(args.weights, 'w') as fw:
    np.savetxt(fw, w_b)