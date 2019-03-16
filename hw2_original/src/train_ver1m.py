import numpy as np
import data_reader
import sys

title, data = data_reader.reader_onehot('data/X_train')
num, dim = data.shape
data_b = np.concatenate((data, np.ones(num).reshape(-1, 1)), 1) #.astype(int)
x = data_b

label, y = data_reader.reader_res('data/Y_train')

w, b = np.zeros(dim), np.array([0])
w_b = np.concatenate((w, b))
# w_b = np.random.random(dim + 1)
iterations, lr = 10000, 8e-15
if len(sys.argv) > 2:
	lr, iterations = eval(sys.argv[1]), eval(sys.argv[2])
elif len(sys.argv) == 2:
    lr = eval(sys.argv[1])
# print("learning rate = ", lr)

sigmoid = lambda z: np.clip(1.0 / (1.0 + np.exp(-z)), 
                            0.00000000000001, 0.99999999999999)
fwb = lambda xxx: sigmoid(xxx.dot(w_b))
res = sigmoid(data_b.dot(w_b))

print('Finish data loading!')

for it in range(iterations):
    loss, grad = 0.0, 0.0
    for dd in range(num):
        loss += -(y[dd]*np.log(fwb(x[dd]))+(1-y[dd])*np.log(1-fwb(x[dd])))
        grad += -(y[dd] - fwb(x[dd]))*x[dd]
    res = sigmoid(data_b.dot(w_b))
    clsed = np.array([1 if ans > 0.5 else 0 for ans in res]).astype(int)
    acc = 1 - sum(abs(y-clsed)) / num
    print("\rTry %7d: loss = %12.7lf, Score = %.5f "%(it + 1, loss / num, acc), end='')
    if acc > 0.84434:
        print(' ==> Simple Baseline Passed??', end='')
    print('\r', end='')
    w_b -= lr * grad
    if it % 10 == 9:
    	with open('weights/model.npy', 'w') as fw:
    		np.savetxt(fw, w_b)

title_t, data_t = data_reader.reader_onehot('data/X_test')
num_t, dim_t = data_t.shape
data_bt = np.concatenate((data_t, np.ones(num_t).reshape(-1, 1)), 1) #.astype(int)

res_t = sigmoid(data_bt.dot(w_b))
classed = [1 if ans > 0.5 else 0 for ans in res_t]

with open(sys.argv[-1], 'w') as fpr:
	fpr.write('id,label\n')
	for nw, dt in enumerate(classed):
		fpr.write(str(nw+1)+','+str(dt)+'\n')
print()
