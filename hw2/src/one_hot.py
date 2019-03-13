import csv

with open('data/train.csv', 'r') as ftr:
	data_tr = [ntr for ntr in csv.reader(ftr)]

with open('data/X_train', 'r') as ftr:
	data_X = [ntr for ntr in csv.reader(ftr)]

x = data_X
t = data_tr

zer = [0] * 106
kkk = dict([(kk, k) for k, kk in enumerate(x[0])][6:])

it = t[1]
res = zer[:]
for xx, yy in enumerate([0, 2, 9, 10, 11, 12]):
	if yy == 9:
		res[xx] = 1 if it[yy] == 'Male' else 0
		continue
	res[xx] = eval(it[yy])


for jj in [1, 3, 5, 6, 7, 8, 13]:
	res[kkk[it[jj]]] = 1
	input(jj)
