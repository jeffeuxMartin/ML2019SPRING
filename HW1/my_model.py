# numpy, scipy, pandas
# os, sys, csv, argparse, time, json, glob
# numpy.linalg.lstsq forbidden
def write(ln):
  with open('my_model.py', 'a') as fw:
    fw.write(ln + '\n')

import csv
import numpy as np

data, month, temp, line = [], [], [], []
with open('train.csv', 'r', encoding='big5', newline='') as csvf:
  rows = csv.reader(csvf)
  for ct, row in enumerate(rows):
    if ct == 0:
      continue
    for entry in row[3:]:
      if entry == 'NR':
        line.append(0)
#        line.append(-1) # for trying
      else:
        line.append(eval(entry))
    temp.append(line); line = []
    if ct % 18 == 0:
      month.append(temp); temp = []
    if ct % (18 * 20) == 0:
      data.append(month); month = []
data = np.transpose(data, (0, 1, 3, 2))
pm25 = np.array([k[9] for i in data for j in i for k in j]).reshape(len(data), len(data[0]), -1)
data = data.reshape(len(data), -1, data.shape[-1])
pm25 = pm25.reshape(len(pm25), -1)

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
import json
import os
if os.path.isfile(weight_f):
  w = np.array(json.load(open(weight_f)), dtype='float64').reshape(-1, 1)
else:
  w = np.random.randn(dim_w_b, 1)
it, lr = 1000000, 3e-10


for i in range(it):
  loss = (sum((Y - X.dot(w)) ** 2) / dim_data) ** 0.5
  if loss > 1e20:
    break
  print("\rTry %6d : Loss = %.4f" % (i + 1, loss), end='\r')
  grad = -2 * X.T.dot(Y - X.dot(w))
  w -= lr * grad
  if i % 1000 == 0:
    json.dump([w_.item() for w_ in w], open(weight_f,'w'))

# data, month, temp, line = [], [], [], []
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
