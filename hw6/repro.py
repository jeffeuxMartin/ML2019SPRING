from glob import glob
import jieba
import numpy as np
import pandas as pd
from keras.models import load_model
from gensim.models import Word2Vec
from multiprocessing import Pool
import sys, os

class argstype:
    def __init__(self):
        self.seq_len = 40
args = argstype()
       
Ys = {}

if __name__ == '__main__':
    jieba.load_userdict('dict.txt.big') 
    test = list(pd.read_csv('test_x.csv')['comment'])
    def tokenize(s): 
        return jieba.lcut(s) 
    P = Pool(processes=4)
    te = P.map(tokenize, test) 
    P.close() 
    P.join()                                                               
    # input('stop')
    # for h5 in glob('*.h5'):
    for h5 in [\
'model_best_20190509_094139__20190509_093754.h5',\
'model_best_20190509_094032__20190509_093745.h5',\
'model_best_20190509_095448__20190509_095147.h5',\
'model_best_20190509_095828__20190509_095536.h5' ]:
        print(h5) 
        vec = 'saved_' + h5.split('__')[1][:-3] + '.model' 
        print(vec) 
        model = load_model(h5) 
        embed = Word2Vec.load(vec) 
       
        w2i = {}
        for n, w in enumerate(embed.wv.vocab):
            w2i[w] = n
        w2i['<UNK>'] = len(w2i)
        w2i['<PAD>'] = len(w2i)

        def turn_indx(sent, pad_sz=None):
            if pad_sz == None: 
                pad_sz = args.seq_len
            return [w2i.get(w) or w2i['<UNK>'] \
                    for w in sent[:pad_sz]] \
                    + [w2i['<PAD>']] * (pad_sz - len(sent))

        tev = np.vstack([turn_indx(s) for s in te])
        Ys[h5] = model.predict(tev)
    Yss = list(Ys.values())
    def rei(out): return np.round(sum(out)/len(out))
    o = rei(Yss)
    with open((str(int(__import__('time').time())))+'.csv', 'w') as ff: ff.write('id,label\n'+''.join([(str(int(n))+','+str(int(w))+'\n') for n,w in enumerate(o)]))
