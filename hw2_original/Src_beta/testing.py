print('\n==================================Testing starts!==================================')
import numpy as np
import data_reader as dt
from argparse import ArgumentParser
import sys

parser = ArgumentParser()
parser.add_argument("dataX", help="Testing one-hot data X.")
parser.add_argument("dataY", help="Output result data Y.")
parser.add_argument("weights", help="The weights of the model.")
parser.add_argument("-f", "--feat", "--iteration", help="Feature scaling used.", 
                    dest="feat", default="")

args = parser.parse_args()
title, data = dt.reader_onehot(args.dataX)
num, dim = data.shape
x = data

# w_b = np.concatenate((w, b))
with open(args.weights, 'r') as f:
    w_b = np.loadtxt(f)
w, b = w_b[:-1], w_b[-1] 


sigmoid = lambda z: np.clip(1.0 / (1.0 + np.exp(-z)), 
                            0.00000000000001, 0.99999999999999)
# res = sigmoid(data_bt.dot(w_b))
res = sigmoid(np.dot(x, np.transpose(w)) + b)
classed = [1 if ans > 0.5 else 0 for ans in res]

with open(args.dataY, 'w') as fpr:
	fpr.write('id,label\n')
	for nw, dt in enumerate(classed):
		fpr.write(str(nw+1)+','+str(dt)+'\n')
print('\n==============================Prediction finished!=================================')