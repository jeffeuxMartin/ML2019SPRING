################### Libraries ####################
import os, sys, json, time, pickle
from multiprocessing import Pool

import numpy as np, pandas as pd, random as rn
from scipy.sparse.linalg import norm
np.random.seed(0); rn.seed(12345)

import jieba
import jieba.analyse
import gensim
from gensim import parsing
from gensim.models import doc2vec, word2vec

from sklearn import feature_extraction
from sklearn.feature_extraction.text \
    import TfidfTransformer, CountVectorizer

################### Parameters ###################
topicFile = "model/JeffTopic.json"
dictFile = "model/dict.txt.big"
contentFile = sys.argv[1] # "url2content.json"
cut_contFile =  "model/Q.json"
tdFile = sys.argv[2] # "TD.csv"
qsFile = sys.argv[3] # "QS_1.csv"

_R = np.load('model/prob_tfidf.npy')
ProbKWV = np.load('model/prob_kwemb.npy')

#################### Ensemble ####################
_Rn =(_R - np.min(_R)) / (np.max(_R) - np.min(_R))
ProbKWV_normalized = (ProbKWV - np.min(ProbKWV))\
    / (np.max(ProbKWV) - np.min(ProbKWV))

_vR = (_Rn * 0.1918345 \
	+ ProbKWV_normalized * 0.2179311) \
    / (0.1918345 + 0.2179311)

_vFF = [(-_vR[_T]).argsort()[:300] \
    for _T in range(20)]
_vFF = np.stack(_vFF)
_vFF += 1

Adjusted = _vR

TD = pd.read_csv(tdFile).to_numpy()
QS = pd.read_csv(qsFile).to_numpy().T[1]
myTD = {}
for _t, _n, _r in TD:
    if myTD.get(_t) is None:
        myTD[_t] = [[], [], [], []]
    myTD[_t][_r].append(int(_n[-6:]) - 1)
for nq, _q in enumerate(QS):
    if not myTD.get(_q) is None:
        a0, a1, a2, a3 = myTD[_q]
        for _nn in a0: Adjusted[nq][_nn] *= 0.1
        for _nn in a1: Adjusted[nq][_nn] += 1
        for _nn in a2: Adjusted[nq][_nn] += 2
        for _nn in a3: Adjusted[nq][_nn] += 3
with open(sys.argv[4], 'w') as f:
    f.write('Query_Index,' + ','.join(
        'Rank_{:03d}'.format(i + 1) \
            for i in range(300)))
    for aA, aB in enumerate(Adjusted):
        f.write('\nq_{:02d}'.format(aA + 1))
        for _tr in ((-aB).argsort() + 1)[:300]:
            f.write(',news_{:06d}'.format(_tr))
