# standard library
import os, csv, sys, argparse
from multiprocessing import Pool

# allowed library
import jieba
import numpy as np
np.random.seed(0)
import pandas as pd
from gensim.models import Word2Vec

# API library
# # PyTorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# # Keras library
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from keras import backend as backend
backend.tensorflow_backend._get_available_gpus()

import time as t
timestamp =t.strftime("%Y%m%d_%H%M%S",t.gmtime(t.time()+8*60*60))

parser = argparse.ArgumentParser()
parser.add_argument('train_X',type=str, help='[Input] Your train_x.csv')
parser.add_argument('train_Y',type=str, help='[Input] Your train_y.csv')
parser.add_argument('test_X',type=str, help='[Input] Your train_y.csv')
parser.add_argument('jieba_lib',type=str, help='[Input] Your jieba dict.txt.big')

parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batch', default=128, type=int)
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--seq_len', default=30, type=int)
parser.add_argument('--word_dim', default=100, type=int)
parser.add_argument('--hidden_dim', default=100, type=int)
parser.add_argument('--dense_dim', default=50, type=int)
parser.add_argument('--wndw', default=3, type=int)
parser.add_argument('--cnt', default=3, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--patience', default=8, type=int)
parser.add_argument('--cont', default="", type=str)
args = parser.parse_args()
# main(args)

def timer(func):
    t = __import__("time").time
    def wrap(*args, **kwargs):
        stime = t(); func(*args, **kwargs)
        print(">>> ", t() - stime, "seconds used...")
    return wrap

class Preprocess():
    @timer
    def __init__(self, 
                 args,
                 train_path=args.train_X, 
                 test_path=args.test_X, 
                 label_path=args.train_Y,
                 dict_path=args.jieba_lib
                ):
        jieba.load_userdict(dict_path)
        self.tr = pd.read_csv(train_path)['comment']
        P = Pool(processes=4); self.tr = P.map(
            self.tokenize, self.tr); P.close(); P.join()
        # self.tr = [jieba.lcut(s) for s in self.tr]
        self.te = pd.read_csv(test_path)['comment']
        P = Pool(processes=4); self.te = P.map(
            self.tokenize, self.te); P.close(); P.join()
        # self.te = [jieba.lcut(s) for s in self.te]
        self.lb = np.array(pd.read_csv(label_path)['label'])
        self.unk = "<UNK>"; self.pad = "<PAD>"
        self.embed_dim, self.wndw_size, self.word_cnt \
         = args.word_dim, args.wndw, args.cnt
        self.save_name = 'saved_{}.model'.format(timestamp)
        self.embed = []
        self.index2word, self.word2index, self.vectors = [], {}, []
        self.trv, self.tev = [], []
        self.trvec, self.tevec = [], []
        self.args = args

    def tokenize(self, sentence): return jieba.lcut(sentence)

    @timer
    def get_embedding(self, load=False):
        print("=== Get embedding")
        # Get Word2vec word embedding
        if load:
            embed = Word2Vec.load(self.save_name)
        else:
            embed = Word2Vec(self.tr + self.te, 
                             size=self.embed_dim, 
                             window=self.wndw_size, 
                             min_count=self.word_cnt, 
                             iter=16, workers=8)
            embed.save(self.save_name)
        if self.args.cont != "":
            embed = Word2Vec.load('saved_{}.model'.format(self.args.cont.split('__')[-1][:-3]))
        # Create word2index dictinonary
        # Create index2word list
        # Create word vector list
        self.index2word = list(embed.wv.vocab.keys())
        self.word2index = {w: n for n, w in enumerate(self.index2word)}
        self.vectors = [embed.wv[w] for w in embed.wv.vocab]
        #     #e.g. self.word2index['魯'] = 1 
        #     #e.g. self.index2word[1] = '魯'
        #     #e.g. self.vectors[1] = '魯' vector

        # Add special tokens
        from torch import empty
        from torch.nn.init import uniform_
        self.word2index[self.unk] = len(self.index2word)
        self.index2word.append(self.unk)
        self.vectors.append(uniform_(empty(1, self.args.word_dim)).numpy())

        self.word2index[self.pad] = len(self.index2word)
        self.index2word.append(self.pad)
        self.vectors.append(uniform_(empty(1, self.args.word_dim)).numpy())

        print("=== total words: {}".format(len(self.vectors)))
        self.vectors = np.vstack(self.vectors)
        # self.embed = embed
        # return self.vectors
        # return self.embed

    # @timer
    def turn_indx(self, sent, pad_sz=None):
        if pad_sz == None: 
            pad_sz = self.args.seq_len
        return [self.word2index.get(w) or self.word2index['<UNK>'] \
                for w in sent[:pad_sz]] \
                + [self.word2index['<PAD>']] * (pad_sz - len(sent))

    @timer
    def get_all_indx(self, pad_sz=None):
        if pad_sz == None:
            pad_sz = self.args.seq_len
        self.trv = np.vstack([self.turn_indx(s) for s in self.tr])
        # self.trvec = np.zeros((len(self.trv), pad_sz, self.args.word_dim))
        # for ns, s in enumerate(self.trv):
        #     for nw, w in enumerate(s):
        #         self.trvec[ns][nw] = self.vectors[w]
        self.tev = np.vstack([self.turn_indx(s) for s in self.te])
        # self.tevec = np.zeros((len(self.tev), pad_sz, self.args.word_dim))
        # for ns, s in enumerate(self.tev):
        #     for nw, w in enumerate(s):
        #         self.tevec[ns][nw] = self.vectors[w]
        # return self.trvec, self.tevec
        

class RNNModel:
    def __init__(self, args, prep):
        import time as t
        self.now =t.strftime("%Y%m%d_%H%M%S",t.gmtime(t.time()+8*60*60))
        self.model = Sequential()
        self.args = args
        self.prep = prep
        self.history = ""

    def build(self):
        print("Building the model...")
        self.model = Sequential()
        self.model.add(Embedding(*(self.prep.vectors.shape), 
            weights=[self.prep.vectors], input_length=args.seq_len, trainable=False))
        self.model.add(LSTM(self.args.hidden_dim, dropout=(1-self.args.dropout), 
                                 recurrent_dropout=(1-self.args.dropout), 
                                 # input_shape=(self.args.seq_len, self.args.word_dim)
                                 ))
        for k in range(self.args.num_layers):
            self.model.add(Dense(self.args.dense_dim, activation='relu')) 
            self.model.add(Dropout(rate=(1-self.args.dropout)))
        self.model.add(Dense(1, activation='sigmoid')) 
        self.model.summary()


    def compile(self):
        if self.args.cont == "":
            self.build()
        else:
            self.model = load_model(self.args.cont)
        print("Compiling the model...@{}__{}".format(self.now, timestamp))
        checkpointer = ModelCheckpoint(
            filepath=('model_best_{}__{}.h5'.format(self.now, timestamp)),
            # {epochs:02d}-{val_acc:.2f}.hdf5
            monitor='val_acc', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.args.patience, verbose=1,
            restore_best_weights=False)

        self.model.compile(loss='binary_crossentropy', 
                           optimizer=Adam(lr=self.args.lr), 
                           metrics=['accuracy']) 
        return checkpointer, early_stopping

    def training(self):
        checkpointer, early_stopping = self.compile()
        print("Training the model...")
        # history = []
        try:
            print('Start')
            self.history = self.model.fit(
                           self.prep.trv, self.prep.lb, 
                           batch_size=self.args.batch,
                           epochs=self.args.epoch, 
                           verbose=1, 
                           validation_split=0.1, 
                           callbacks=[checkpointer, 
                                      early_stopping])
        except KeyboardInterrupt:
            print("Early cut!")
        finally:
            print("Finally Saving the model...")
            self.model.save('model_final_{}__{}.h5'.format(self.now, timestamp))
            # return history

    def Save_pred(self, Y):
        print("Saving the prediction as", 
              'prediction_best_{}__{}.csv'.format(self.now, timestamp), "...")
        import json
        # with open('value_best_{}__{}.csv'.format(self.now, timestamp), 'w') as f: 
            # json.dump(Y, f)
        np.save('value_best_{}__{}'.format(self.now, timestamp), Y)
        with open('prediction_best_{}__{}.csv'.format(self.now, timestamp), 'w') as f: 
            f.write('id,label\n') 
            for j in range(len(Y)): 
                f.write(str(j)+',') 
                if Y[j] >= 0.5: 
                    f.write(str(1)) 
                else: 
                    f.write(str(0)) 
                f.write('\n') 

    def predicting(self, model_type='best'):
        print("Predicting...")
        self.model = load_model('model_{}_{}__{}.h5'.format(model_type, self.now, timestamp))
        Y = self.model.predict(self.prep.tev)
        self.Save_pred(Y)
        return Y


if __name__ == "__main__":
    p = Preprocess(args)
    p.get_embedding()
    # p.vectors = np.vstack(p.vectors)
    p.get_all_indx()
    
    nn = RNNModel(args, p)
    # nn.compile()
    nn.training()
    try:
        nn.predicting()
    except OSError:
        nn.predicting('final')
