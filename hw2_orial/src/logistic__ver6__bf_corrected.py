#### TODO ####
# Early Stopping

import numpy as np
import pandas as pd
from argparse import ArgumentParser
import os

## arguments
parser = ArgumentParser()
parser.add_argument("dataX", help="Training one-hot data X.")
parser.add_argument("dataY", help="Training result data Y.")
parser.add_argument("-n", "-k", "--n_fold", "--k_fold", help="Number of cross validations.", dest="folds", default="3", type=int)
parser.add_argument("-i", "-it", "--iter", "--iteration", help="Number of iterations.", dest="iterations", default="100", type=int)
parser.add_argument("-lr", "--lr", help="Learning rate.", dest="lr", default="1e-4")
parser.add_argument("-b", "-bt", "--batch", help="Batch size.", dest="bat", default="1000", type=int)
parser.add_argument("-f", "--feat", help="Feature scaling used.", dest="feat", default="weights/feat_scale.npy")
parser.add_argument("weights", help="The weights of the model.")
parser.add_argument("-rgl", "--lmbda", help="Regularization term.", dest="rgl", default="12")
parser.add_argument("-eps1", "--epsilon1", help="Adagrad epsilon term inside the sqrt.", dest="eps1", default="1e-8", type=float)
parser.add_argument("-eps2", "--epsilon2", help="Adagrad epsilon term outside the sqrt.", dest="eps2", default="0", type=float)
parser.add_argument("-b1", "--beta1", help="Adam momentum term 1.", dest="beta1", default="0.9", type=float)
parser.add_argument("-b2", "--beta2", help="Adam momentum term 2.", dest="beta2", default="0.9909", type=float)
args = parser.parse_args()

## Reading data
data = np.array(pd.read_csv(args.dataX))
num, dim = data.shape
 ## normalization
mean_, std_ = data.mean(0), data.std(0)
np.savetxt(args.feat, np.array([mean_, std_]))

## training X
x_ori = data.copy()
data = (data - mean_) / std_
data_b = np.concatenate((data, np.ones(num).reshape(-1, 1)), 1) #.astype(int)
x_all = data_b
## training Y
y_all = np.array(pd.read_csv(args.dataY)).reshape(-1) 

## parameters
iterations, lr = int(args.iterations), eval(args.lr)
lmbda, eps1, eps2 = eval(args.rgl), args.eps1, args.eps2
beta1, beta2 = args.beta1, args.beta2
batch = args.bat
folds = args.folds

sigmoid = lambda z: np.clip(1.0 / (1.0 + np.exp(-z)), 0.00000000000001, 0.99999999999999)

## weights and bias
w, b = np.zeros((dim,)), np.zeros((1,))
w_b = np.concatenate((w.copy(), b.copy()))
WBS = [w_b.copy() for k in range(folds)]
wfn = args.weights.split('.')
if len(wfn) == 2:
    wfn, wfnext = wfn
wfn = wfn.split('/')
if len(wfn) == 2:
    wfndir, wfn = wfn
while os.path.isdir(wfndir + '/' + wfn):
    wfn = '_' + wfn
os.mkdir(wfndir + '/' + wfn)

sz = num // folds
for k in range(folds):
    x_ft = x_all[k * sz : (k+1) * sz]
    x_fv = np.concatenate((x_all[:k * sz], x_all[(k+1) * sz:]), 0)
    y_ft = y_all[k * sz : (k+1) * sz]
    y_fv = np.concatenate((y_all[:k * sz], y_all[(k+1) * sz:]), 0)

    w_b = WBS[k]
    ## functions and helpers
    res = sigmoid(data_b.dot(w_b))
    prev_grav = np.zeros((dim + 1,))
    prev_momt = np.zeros((dim + 1,))

    icon = 100
    # icon = shutil.get_terminal_size().columns - 112
    for it in range(iterations):
        try:
            if batch != 0:
                biters = num // batch
            else:
                biters = 1
                batch = num
            for bt in range(biters):
                print('\r', end='')
                x = x_ft[(bt * batch):((bt + 1) * batch)]
                y = y_ft[(bt * batch):((bt + 1) * batch)]
                # print(x.shape, y.shape, end='')

                loss = -sum(y*(np.log(sigmoid(x.dot(w_b)))) + (1-y)*(np.log(1-sigmoid(x.dot(w_b))))) + lmbda * sum(w_b[:-1]**2)
                grad = -(y-sigmoid(x.dot(w_b))).dot(x) + 2 * lmbda * np.concatenate((w_b[:-1], np.array([0])), 0)
                prev_momt = beta1 * prev_momt + (1 - beta1) * grad
                prev_grav = beta2 * prev_grav + (1 - beta2) * grad ** 2
                momt = prev_momt / (1 - beta1)
                grav = prev_grav / (1 - beta2)

                print("Fold {0:2d} / {1:2d} : ".format(k + 1, folds), end='')
                print("Iteration {0:4d} / {1:4d} : ".format(it + 1, iterations), end='')
                print("Batch {0:4d} / {1:4d} : ".format(bt + 1, biters), end='')
                print("loss = {:10.7f}".format(loss / biters), end='')

                res = sigmoid(x_ft.dot(w_b))
                clsed = np.array([1 if ans > 0.5 else 0 for ans in res]).astype(int)
                acc = 1 - sum(abs(y_ft-clsed)) / num        
                print(", Score = {:.5f}  ".format(acc), end='')

                res = sigmoid(x_fv.dot(w_b))
                clsed = np.array([1 if ans > 0.5 else 0 for ans in res]).astype(int)
                acc = 1 - sum(abs(y_fv-clsed)) / num        
                print(", Score = {:.5f}  ".format(acc), end='')

                print('\r', end='')
                w_b -= lr * momt / (np.sqrt(prev_grav + eps1) + eps2)
                
            # print()
            with open(wfndir + '/' + wfn + '/' + wfn + '_' + str(k) + '.' + wfnext, 'w') as fw:
                np.savetxt(fw, w_b)
        except KeyboardInterrupt:
            break
    print('\n')

for k in range(folds):
    with open(wfndir + '/' + wfn + '/' + wfn + '_' + str(k) + '.' + wfnext, 'w') as fw:
        np.savetxt(fw, WBS[k])

## validation
grades = np.zeros((folds, folds))
avegrad = np.zeros((folds,))
for k in range(folds):
    # x_ft = x_all[k * sz : (k+1) * sz]
    x_fv = np.concatenate((x_all[:k * sz], x_all[(k+1) * sz:]), 0)
    # y_ft = y_all[k * sz : (k+1) * sz]
    y_fv = np.concatenate((y_all[:k * sz], y_all[(k+1) * sz:]), 0)
    
    with open(wfndir + '/' + wfn + '/' + wfn + '_result.log', 'a') as fw:
        for n, w_b in enumerate(WBS):
            res = sigmoid(x_fv.dot(w_b))
            clsed = np.array([1 if ans > 0.5 else 0 for ans in res]).astype(int)
            acc = 1 - sum(abs(y_fv-clsed)) / num        
            fw.write("Model {} on fold {}, Score = {:.5f}  \n".format(n, k, acc))
            grades[n, k] = acc
for k in range(folds):
    avegrad[k] = sum(grades[k]) / folds
avegraddict = {m: n for n, m in enumerate(avegrad)}
Highest = max(avegraddict)
res = 'The model {} has the highest score: {:.5f}.'.format(avegraddict[Highest], Highest)
print(res)
with open(wfndir + '/' + wfn + '/' + wfn + '_result.log', 'a') as fw:
    fw.write('\n'+res)
with open(wfndir + '/' + wfn + '/' + wfn + '.' + wfnext, 'w') as fw:
    np.savetxt(fw, WBS[avegraddict[Highest]])