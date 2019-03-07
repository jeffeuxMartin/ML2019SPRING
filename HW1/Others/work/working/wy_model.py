# numpy, scipy, pandas
# os, sys, csv, argparse, time, json, glob
# numpy.linalg.lstsq forbidden
"""
def write(ln):
  with open('my_model.py', 'a') as fw:
    fw.write(ln + '\n')
"""
# Standard Python Libraries
import csv
import os
import sys
import json
import time
# Allowed Library
import numpy as np

beginning_t = time.time()
print(time.ctime())

with open('train.csv', 'r', encoding='big5', newline='') as f:   
    data = np.array([[eval(data) if data != 'NR' else 0 
       for data in r[3:]] for r in ([r for r in csv.reader(f)][1:])])
    data_c = data.reshape(12, 20, 18, 24)                           \
                           .transpose(2, 0, 1, 3).reshape(18, 12, -1)

lowest = np.array([feat.min() for feat in data_c])
rangin = np.array([feat.max() - feat.min() for feat in data_c])
s_mean = np.array([feat.mean() for feat in data_c])
s_stdd = np.array([feat.std() for feat in data_c])

feat_scal_u, feat_scal_s = False, False
if feat_scal_u:
    data_c = np.array([(feat - lowest[n]) / rangin[n]               \
                                   for n, feat in enumerate(data_c)])
elif feat_scal_s:
    data_c = np.array([(feat - s_mean[n]) / s_stdd[n]               \
                                   for n, feat in enumerate(data_c)])

pm25 = data_c[9]
data = data_c.reshape(18, 12, 20, 24)                               \
                          .transpose(1, 2, 3, 0).reshape(12, 480, 18)

# data with 12 x 480 x 18
# pm25 with 12 x 480

feat_x, feat_y = [], []
for month in data:
  # month with 480 x 18
  for hour in range(480 - 10 + 1): # 0 ~ 469
    feat_x.append(month[hour:(hour + 10 - 1)])
    feat_y.append(month[hour + 10 - 1][9])
feat_x, feat_y = np.array(feat_x).reshape(len(feat_x), -1), np.array(feat_y).reshape(-1, 1)
# len = 5652
dim_data, dim_w_b = feat_x.shape; dim_w_b += 1
X, Y = np.concatenate((feat_x, np.ones((len(feat_x), 1))), axis = 1), feat_y

weight_f = './weight.json'

if os.path.isfile(weight_f):
  w = np.array(json.load(open(weight_f)), dtype='float64').reshape(-1, 1)
else:
  w = np.zeros((dim_w_b, 1))
w = np.zeros((dim_w_b, 1))
it, lr = 10000000000000, eval(sys.argv[1]) if len(sys.argv) >= 2 else 300
#it, lr = 10, 3e-10
prev_gra = np.zeros((18 * 9 + 1, 1)) # adagrad
prev_loss = 10000000

#printfreq = int(sys.argv[1]) if len(sys.argv) >= 2 else 13
#printfreq_ = printfreq - 1
rec = 0

print("learning rate =", lr)
for i in range(it):
  loss = (sum((Y - X.dot(w)) ** 2) / dim_data) ** 0.5
  if loss > prev_loss + 0.1:
  	break
  elif loss - prev_loss > 0.0001:
  	input("\nKeep going?")
  if loss > 1e20:
    break
#  if i % printfreq == printfreq_:
  print("iteration %14d / %14d : Loss = %.4f               " % (i + 1, it, loss), end='\r')
  if i % 2500 == 2500 - 1:
    print('\n                    ^^^^         every 2500 iterations        ', time.ctime(), ' Taking %15.4lf seconds' % (time.time() - beginning_t)); continue
  if loss < 8 and rec == 0: print('\n                                                   ^ loss == 8', time.ctime(), ' Taking %15.4lf seconds' % (time.time() - beginning_t)); rec += 1; continue
  if loss < 7 and rec == 1: print('\n                                                   ^ loss == 7', time.ctime(), ' Taking %15.4lf seconds' % (time.time() - beginning_t)); rec += 1; continue
  if loss < 6 and rec == 2: print('\n                                                   ^ loss == 6', time.ctime(), ' Taking %15.4lf seconds' % (time.time() - beginning_t)); rec += 1; continue
  if loss < 5 and rec == 3: print('\n                                                   ^ loss == 5', time.ctime(), ' Taking %15.4lf seconds' % (time.time() - beginning_t)); rec += 1; continue
  print('\r', end='')
#break
  grad = -2 * X.T.dot(Y - X.dot(w))
  prev_gra += grad ** 2
  ada = np.sqrt(prev_gra)
  w -= lr * grad / ada
  if i % 500 == 0:
    json.dump([w_.item() for w_ in w], open(weight_f,'w'))

print()

tdata, tmp = [], []
with open('test.csv', 'r', encoding='big5', newline='') as csvf2:
  rows2 = csv.reader(csvf2)
  for ct2, row2 in enumerate(rows2):
    tmp.append([eval(item_) if item_ != 'NR' else 0 for item_ in row2[2:]])
    if ct2 % 18 == 18 - 1:
      tdata.append(tmp); tmp = []
tdata = np.array(tdata)
tdata = np.transpose(tdata, (0, 2, 1)).reshape(len(tdata), -1)
tX = np.concatenate((tdata, np.ones((len(tdata), 1))), axis = 1)
res = tX.dot(w)

with open('prediction.csv', 'w', newline='') as csvfile:
  writer = csv.writer(csvfile)

  writer.writerow(['id', 'value'])
  for ct3, row3 in enumerate(res):
    writer.writerow(['id_'+str(ct3), row3.item()])
print("Prediction done!")
print('\b' + time.ctime(), ' Taking %15.4lf seconds' % (time.time() - beginning_t))
