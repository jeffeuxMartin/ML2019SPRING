import numpy as np
from sklearn import ensemble
import pandas as pd
import sys
import os
import json

Xori = np.array(pd.read_csv(sys.argv[3]))
dtmean, dtstd = Xori.mean(0), Xori.std(0)
Xori = (Xori - dtmean) / dtstd

X = Xori[:30000]
Yori = np.array(pd.read_csv(sys.argv[4])).reshape((-1,))
Y = Yori[:30000]
lr = 0.08
n_est = 500

if len(sys.argv) >= 3:
    lr=eval(sys.argv[1]) 
    n_est=int(sys.argv[2])
elif len(sys.argv) == 2:
    lr=eval(sys.argv[1])
    n_est=500
else:
    lr=0.08
    n_est=500
clf = ensemble.GradientBoostingClassifier(learning_rate=lr, n_estimators=n_est, verbose=1)
clf.fit(X, Y)

tX = np.array(pd.read_csv(sys.argv[5]))
tX = (tX - dtmean) / dtstd

tY = clf.predict(tX)
print()
print(sum(tY)/len(tY)*100, end='%\n')

Xval = Xori[30000:]
Yval = Yori[30000:]
Yprval = clf.predict(Xval)
print((1 - sum(abs(Yval - Yprval)) / len(Yval))*100, end='%\n')

pred_name = 'sklearn_GBC_pred' + '__lr_' + str(lr) + '__nest_' + str(n_est) + "__"
idno = 1
while os.path.isdir(pred_name + "{:02d}".format(idno)):
	idno += 1
pred_name = pred_name + "{:02d}".format(idno)
os.mkdir(pred_name)

with open(pred_name + '/' + pred_name + '.csv', 'w') as fpr:
    fpr.write('id,label\n')
    for nw, dt in enumerate(tY):
        fpr.write(str(nw+1)+','+str(dt)+'\n')
with open(pred_name + '/' + pred_name + '_params.json', 'w') as fpr:
	json.dump(clf.get_params(), fpr)

# 19.366132301455686 %
# 19.47669062096923%
# 0.08 500
