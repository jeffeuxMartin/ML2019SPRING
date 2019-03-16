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
label, y = data_reader.reader_res(args.dataY)

x = np.concatenate((data, np.ones(num).reshape(-1, 1)), 1) #.astype(int)

w, b = np.zeros(dim), np.array([0])
w_b = np.concatenate((w, b))
# w_b = np.random.random(dim + 1)

iterations, lr = int(args.iterations), eval(args.lr)
# sigmoid(x.dot(w_b))
# sigmoid(data.dot(w) + b)

"""for j in range(iterations):
    pass"""

ct_0 = ct_1 = 0; mean0 = mean1 = np.zeros(dim, )
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
Var = (ct_0 * Var0 + ct_1 * Var1) / num

"""data0, data1 = [], []
for n in range(num):
    if y[n] == 1:
        data1.append(data[n])
    else:
        data0.append(data[n])
data0, data1 = np.array(data0), np.array(data1)
mean0, mean1 = np.mean(data0, 1), np.mean(data1, 1)"""

PC0, PC1 = ct_0 / num, ct_1 / num
antiVar = np.linalg.inv(Var)


def probB(v_indx, _x, _y, _mean0, _mean1, _antiV):
    mean_c = _mean0 if _y[v_indx] == 0 else _mean1
    v_1 = (_x[v_indx] - mean_c).reshape(-1, 1)
    res_tmp = v_1.T.dot(_antiV).dot(v_1)
    return -res_tmp / 2

# probB(0, data, y, mean0, mean1, antiVar)
# -1 / 2 * (((v - mean0).reshape(1, -1)).dot(antiVar)).dot(((v - mean0).reshape(-1, 1)))

sgn, lgdet = np.linalg.slogdet(Var)
'''
In [185]: probB(0, data, y, mean0, mean1, antiVar)                                      
Out[185]: array([[-142831.32939426]])

In [186]: probB(0, data, y, mean0, mean1, antiVar) - lgdet / 2                          
Out[186]: array([[-142065.00449776]])

In [187]: -106 * np.log(2*np.pi)+ (probB(0, data, y, mean0, mean1, antiVar) - lgdet / 2)
     ...:                                                                               
Out[187]: array([[-142259.8194668]])

In [188]: np.exp(-106 * np.log(2*np.pi)+ (probB(0, data, y, mean0, mean1, antiVar) - lgd
     ...: et / 2))                                                                      
Out[188]: array([[0.]])
'''
ans0 = [np.exp(-dim / 2 * np.log(2*np.pi)+ (probB(tr, data, y, mean0, mean1, antiVar) - lgdet / 2)) for tr in range(num)]
    