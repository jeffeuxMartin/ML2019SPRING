import csv
import numpy as np

# 1, 1, 1, 1, 1, <6>, 9, 16, 7, 15, 6, 5, 42
def slicer(lst, *num):
    tot = len(lst)
    acc = 0
    res = []
    for j in num:
        res.append(lst[acc: acc + j])
        acc += j
    res.append(lst[acc: tot])
    return res

def onehotslc(lst):
#    return slicer(lst, 1, 1, 1, 1, 1, 1, 9, 16, 7, 15, 6, 5, 42)
   return slicer(lst, 6, 9, 16, 7, 15, 6, 5)

def reader_raw(fname):
    with open(fname, 'r') as ftr:
        data_tr = [ntr for ntr in csv.reader(ftr)]
    return data_tr

def reader_onehot(fname):
    with open(fname, 'r') as ftr:
        data_X = [ntr for ntr in csv.reader(ftr)]
    for n, ntr in enumerate(data_X[1:]):
        data_X[n + 1] = [eval(nn) for nn in ntr]
    return data_X[0], np.array(data_X[1:])

def reader_res(fname):
    with open(fname, 'r') as ftr:
        data_Y = [ntr for ntr in csv.reader(ftr)]
    return data_Y[0], np.array([int(nn[0]) for nn in data_Y[1:]])