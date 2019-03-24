import numpy as np
from sklearn import ensemble
import pandas as pd
import sys
import os
import json
import pickle

with open(sys.argv[2], 'rb') as f:
	clf = pickle.load(f)

data = np.array(pd.read_csv(sys.argv[1]))

mean_, std_ = np.loadtxt(sys.argv[3])
x = (data - mean_) / std_

tY = clf.predict(x)

with open(sys.argv[4], 'w') as fpr:
    fpr.write('id,label\n')
    for nw, dt in enumerate(tY):
        fpr.write(str(nw+1)+','+str(dt)+'\n')