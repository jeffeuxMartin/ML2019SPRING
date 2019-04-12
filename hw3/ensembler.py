from glob import glob as glob
import numpy as np
from pandas import read_csv
import sys
import argparse

seed = 0
np.random.seed(seed)

parser = argparse.ArgumentParser(description='Process data filenames.')
parser.add_argument('--results', '-rs', '--rs', dest='rs', help='', default='.')
parser.add_argument('--outputs', '-out', '--out', dest='out', help='', default='ensembledPrediction.csv')
args = parser.parse_args()

labels = [np.array(read_csv(csv)['label']) for csv in glob(args.rs + '/*prediction_*.csv')]
print('>>>', len(labels), 'files ensembled.')
def one_hot(label): r = np.zeros(7); r[label] = 1; return r 
voting_results = sum([np.array([one_hot(i) for i in q]) for q in labels])
results = [np.argmax(q) for q in voting_results]

def now_name(): import time; return time.strftime("%m%d_%H%M%S", time.gmtime(time.time() + 8*60*60)) 
output_file_path = args.out

with open(output_file_path, 'w') as f:
        f.write('id,label\n')
        for _id, _label in enumerate(results):
            f.write(str(_id) + ',' + str(_label) + '\n')
print('>>> Ensembling predictions done!')
print(output_file_path)