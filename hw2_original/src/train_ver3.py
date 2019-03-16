import numpy as np
from argparse import ArgumentParser
import data_reader

parser = ArgumentParser()
parser.add_argument("dataX", help="Training one-hot data X.")
parser.add_argument("dataY", help="Training result data Y.")
# parser.add_argument('--foo', help='foo help')
# parser.add_argument('--text', '-t', type=str, required=True, help='Text for program')
parser.add_argument("-i", "--iter", "--iteration", help="Number of iterations.", 
                    dest="iterations", default="10000")
parser.add_argument("-lr", "--lr", help="Learning rate.", 
                    dest="lr", default="8e-15")
# parser.add_argument("-o", "--optional-arg", help="optional argument", 
#                     dest="opt", default="default")
# parser.print_help()
args = parser.parse_args()

title, data = data_reader.reader_onehot(args.dataX)
num, dim = data.shape
label, Y = data_reader.reader_res(args.dataY)
X = data
# x = np.concatenate((data, np.ones(num).reshape(-1, 1)), 1) #.astype(int)

w, b = np.zeros(dim), np.array([0])
# w_b = np.concatenate((w, b))
# w_b = np.random.random(dim + 1)

iterations, lr = int(args.iterations), eval(args.lr)
# sigmoid(x.dot(w_b))
# sigmoid(data.dot(w) + b)

"""for j in range(iterations):
    pass"""

# batch_sz = 25

# z = np.dot(X, np.transpose(w)) + b
# y = sigmoid(z)
# cross_entropy = -(np.dot(Y, np.log(y))) + np.dot((1 - Y), np.log(1 - y)))
# w_grad = np.mean(-1 * X * (Y - y).reshape((batch_sz, 1)), axis = 0)
# w -= lr * w_grad
# b_grad = np.mean(-1 * (Y - y))
# b -= lr * b_grad

ct_0 = 0; ct_1 = 0
mean0 = np.zeros((dim, ))
mean1 = np.zeros((dim, ))
for n in range(num):
    if y[n] == 1:
        ct_1 += 1; mean1 += data[n]
    if y[n] == 0:
        ct_0 += 1; mean0 += data[n]
mean0 /= ct_0; mean1 /= ct_1

Var0 = Var1 = np.zeros((dim, dim))
Cov = lambda v: v.reshape(-1, 1) * v.reshape(1, -1)
for n in range(num):
    if y[n] == 1:
        Var1 += Cov(data[n] - mean1)
    if y[n] == 0:
        Var0 += Cov(data[n] - mean0)
Var0 /= ct_0; Var1 /= ct_1
Var = (float(ct_0) * Var0 + float(ct_1) * Var1) / num

# def predict()