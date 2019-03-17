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
parser.add_argument("-i", "-it", "--iter", "--iteration", 
                       help="Number of iterations.", 
                         dest="iterations", default="1000", type=int)
parser.add_argument("-lr", "--lr", help="Learning rate.", 
                                           dest="lr", default="2e-1")
parser.add_argument("-f", "--feat", help="Feature scaling used.", 
                       dest="feat", default="weights/feat_scale.npy")
parser.add_argument("weights", help="The weights of the model.")
parser.add_argument("-rgl", "--lmbda", help="Regularization term.", 
                                            dest="rgl", default="12")
parser.add_argument("-eps1", "--epsilon1", 
                        help="Adagrad epsilon term inside the sqrt.", 
                                dest="eps1", default="1e-8", type=float)
parser.add_argument("-eps2", "--epsilon2", 
                       help="Adagrad epsilon term outside the sqrt.", 
                             dest="eps2", default="0", type=float)
parser.add_argument("-b1", "--beta1", help="Adam momentum term 1.", 
                             dest="beta1", default="0.9", type=float)
parser.add_argument("-b2", "--beta2", help="Adam momentum term 2.", 
                             dest="beta2", default="0.999", type=float)
args = parser.parse_args()
data = np.array(pd.read_csv(args.dataX))
num, dim = data.shape
mean_, std_ = data.mean(0), data.std(0)
np.savetxt(args.feat, np.array([mean_, std_]))

x_ori = data.copy()
data = (data - mean_) / std_
data_b = np.concatenate(
                (data, np.ones(num).reshape(-1, 1)), 1) #.astype(int)
x = data_b

y = np.array(pd.read_csv(args.dataY)).reshape(-1) 
w, b = np.zeros((dim,)), np.zeros((1,))
w_b = np.concatenate((w, b))

iterations, lr = int(args.iterations), eval(args.lr)
lmbda, eps1, eps2 = eval(args.rgl), args.eps1, args.eps2
beta1, beta2 = args.beta1, args.beta2


sigmoid = lambda z: np.clip(1.0 / (1.0 + np.exp(-z)), 
                            0.00000000000001, 0.99999999999999)
res = sigmoid(data_b.dot(w_b))
prev_grav = np.zeros((dim + 1,))
prev_momt = np.zeros((dim + 1,))

# icon = shutil.get_terminal_size().columns - 112
icon = 50
for it in range(iterations):
    try:
        loss = -sum(y*(np.log(sigmoid(x.dot(w_b))))                 \
                      + (1-y)*(np.log(1-sigmoid(x.dot(w_b)))))      \
                                           + lmbda * sum(w_b[:-1]**2)
        grad = -(y-sigmoid(x.dot(w_b))).dot(x)                      \
           + 2 * lmbda * np.concatenate((w_b[:-1], np.array([0])), 0)
        prev_momt = beta1 * prev_momt + (1 - beta1) * grad
        prev_grav = beta2 * prev_grav + (1 - beta2) * grad ** 2
        momt = prev_momt / (1 - beta1)
        grav = prev_grav / (1 - beta2)

        print("\rIteration {0:6d} / {1:6d} : ".format(
            it + 1, iterations), end='')
        # print("lr = {:7.6e}, ".format(
        #    sum((lr / (np.sqrt(prev_grav + eps1) + eps2))**2)), end='')
        print("loss = {:8.7f}".format(loss / num), end='')

        # res = sigmoid(data_b.dot(w_b))
        # clsed = np.array([1 if ans > 0.5 else 0 
        #                                  for ans in res]).astype(int)
        # acc = 1 - sum(abs(y-clsed)) / num        
        # print(", Score = {:.5f}  ".format(acc), end='')

        # if it == iterations - 1:
        #     it += (icon - 1) - (it % icon)
        # print("[" + (it % icon - 1)*'=', end='')
        # if (it % icon != 0) and (it % icon != (icon - 1)):
        #     print('>', end='')
        # print((icon-(it % icon - 1) - 3)*'.'+']', end='')
        # if acc > 0.85565:
        #     print(' ==> Strong Baseline Passed??', end='')
        # elif acc > 0.84434:
        #     print(' ==> Simple Baseline Passed??', end='')
        print('\r', end='')


        w_b -= lr * momt / (np.sqrt(prev_grav + eps1) + eps2)
        if it % icon == (icon - 1):
            print()
            with open(args.weights, 'w') as fw:
                np.savetxt(fw, w_b)
    except KeyboardInterrupt:
        break
print('\n')

with open(args.weights, 'w') as fw:
    np.savetxt(fw, w_b)