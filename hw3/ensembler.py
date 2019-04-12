from glob import glob as glob
import numpy as np
from pandas import read_csv
import sys

seed = 0
np.random.seed(seed)

labels = [np.array(read_csv(csv)['label']) for csv in glob('./*prediction_*.csv')]
print('>>>', len(labels), 'files ensembled.')
def one_hot(label): r = np.zeros(7); r[label] = 1; return r 
voting_results = sum([np.array([one_hot(i) for i in q]) for q in labels])
results = [np.argmax(q) for q in voting_results]

def now_name(): import time; return time.strftime("%m%d_%H%M%S", time.gmtime(time.time() + 8*60*60)) 
output_file_path = './ensembledPrediction_%s.csv'%now_name() if len(sys.argv) < 2 else sys.argv[1]

with open(output_file_path, 'w') as f:
        f.write('id,label\n')
        for _id, _label in enumerate(results):
            f.write(str(_id) + ',' + str(_label) + '\n')
print('>>> Ensembling predictions done!')
print(output_file_path)