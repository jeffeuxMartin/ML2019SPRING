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
topicFile = "model/JeffTopic.json"
dictFile = "model/dict.txt.big"
contentFile = sys.argv[1] # "url2content.json"
cut_contFile =  "model/Q.json"
tdFile = sys.argv[2] # "TD.csv"
qsFile = sys.argv[3] # "QS_1.csv"
w2vModelFile = "model/word2vec1500.model"

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

# ############### Data Preprocessing ###############
# def tokenize(sentence): 
#     return jieba.lcut(sentence)

# if os.path.isfile(cut_contFile):
#     with open(cut_contFile) as f:
#         cut_dic_content = json.load(f)
# else:
#     cut_dic_content = [jieba.lcut(_s) \
#         for _s in dic_content]
# cut_TDQuery = [jieba.lcut(_s) for _s in TDQuery]
# cut_QSQuery = [jieba.lcut(_s) for _s in QSQuery]
# cut_topic = [jieba.lcut(_s) \
#     for _s in dic_topic.values()]
# print(((len(NewsDict)),
#     (len(dic_content), len(TDQuery), len(QSQuery))
# ))
# CutDict = cut_dic_content \
#      + cut_TDQuery + cut_QSQuery
# RejoinedDict = [' '.join(sentence) \
#     for sentence in CutDict]

############## KeyWord weighted W2V ##############
def extracter(doc):
    return jieba.analyse.extract_tags(
        doc, topK=100, withWeight=True)

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

ProbKWV = ((QSembed.dot(QK.T) / np.linalg.norm(
   QK, axis=1)).T / np.linalg.norm(QSembed, axis=1)).T
# _FF2 = (-ProbKWV).argsort(1).T[:300].T + 1
np.save('model/prob_kwemb', arr=ProbKWV)

