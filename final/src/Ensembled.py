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
# from sklearn.neighbors import NearestNeighbors

################### Parameters ###################
topicFile = "../model/JeffTopic.json"
dictFile = "../model/dict.txt.big"
contentFile = "url2content.json"
cut_contFile = sys.argv[1] # "Q.json"
tdFile = sys.argv[2] # "TD.csv"
qsFile = sys.argv[3] # "QS_1.csv"
w2vModelFile = "../model/word2vec1500.model"

################# Loading Files ##################
jieba.set_dictionary(dictFile)
jieba.load_userdict(dictFile)
with open(topicFile, 'r') as f:
    dic_topic = json.load(f)
with open(contentFile, 'r') as f:
    dic_content = list(json.load(f).values())
TD, QS = pd.read_csv(tdFile), pd.read_csv(qsFile)
TDQuery, QSQuery = \
    TD.Query.to_list(), QS.Query.to_list()
NewsDict = dic_content + TDQuery + QSQuery

############### Data Preprocessing ###############
if os.path.isfile(cut_contFile):
    with open(cut_contFile) as f:
        cut_dic_content = json.load(f)
else:
    cut_dic_content = [jieba.lcut(_s) \
        for _s in dic_content]
cut_TDQuery = [jieba.lcut(_s) for _s in TDQuery]
cut_QSQuery = [jieba.lcut(_s) for _s in QSQuery]
cut_topic = [jieba.lcut(_s) \
    for _s in dic_topic.values()]
print(((len(NewsDict)),
    (len(dic_content), len(TDQuery), len(QSQuery))
))
CutDict = cut_dic_content \
     + cut_TDQuery + cut_QSQuery
RejoinedDict = [' '.join(sentence) \
    for sentence in CutDict]

############### Scikit-learn tools ###############
transformer = TfidfTransformer()
cv = CountVectorizer(max_features=None)

############# Self-defined functions #############
def tokenize(sentence): 
    return jieba.lcut(sentence)
def extracter(doc):
    return jieba.analyse.extract_tags(
        doc, topK=100, withWeight=True)

################### TF-IDF BOW ###################
tfidf = transformer.fit_transform(
    cv.fit_transform(RejoinedDict))
#words = cv.get_feature_names()
# 14:55:57

_R = []
for _C in range(20):
    _A = -(20-_C)
    _R.append(
       (tfidf[_A].dot(tfidf[:100000].T).toarray()\
      / norm(tfidf[_A]) / norm(tfidf[:100000].T)))
_R = np.concatenate(_R, 0)

_FF = [(-_R[_T]).argsort()[:300] \
    for _T in range(20)]
_FF = np.stack(_FF)
_FF += 1

############## KeyWord weighted W2V ##############
with Pool(processes=2) as P:
    KeywordNewsDict = P.map(extracter, NewsDict)
P.join()

wvmodel =gensim.models.Word2Vec.load(w2vModelFile)
_vecmean = np.mean(wvmodel.wv.vectors, 0)

def kwaveW2v(_K):
    if len(_K) == 0:
        return _vecmean
    _A = []
    for _kw, prob in _K:
        try:
            _A.append(wvmodel.wv[_kw] * prob)
        except:
            _A.append(_vecmean * prob)
    return sum(_A) \
        / sum(np.array([eval(_n) \
        for _n in np.array(_K).T[1]]))

with Pool(processes=2) as P:
    QK = P.map(kwaveW2v, KeywordNewsDict)
P.join()
QK = np.vstack(QK)[:100000]
QSembed = [kwaveW2v(extracter(_qs)) \
    for _qs in QSQuery]
QSembed = np.vstack(QSembed)

___a = ((QSembed.dot(QK.T) / np.linalg.norm(\
   QK, axis=1)).T / np.linalg.norm(QSembed, axis=1)).T
_FF2 = (-___a).argsort(1).T[:300].T + 1

#################### Ensemble ####################
_Rn =(_R - np.min(_R)) / (np.max(_R) - np.min(_R))
___an = (___a - np.min(___a))\
    / (np.max(___a) - np.min(___a))

_vR = (_Rn * 0.1918345 + ___an * 0.2179311) \
    / (0.1918345 + 0.2179311)

_vFF = [(-_vR[_T]).argsort()[:300] \
    for _T in range(20)]
_vFF = np.stack(_vFF)
_vFF += 1

# Adjusted = np.zeros((20, 100000))
# for i in prob.to_numpy():
#     row = int(i[0][-2:])-1
#     for nj, j in enumerate(i[1:]):
#         val = int(j[-6:]) - 1
#         Adjusted[row][val] = 1 / (nj + 1)
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
