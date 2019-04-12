from keras.models import load_model 
import numpy as np
from glob import glob as glob
import time
import pandas as pd                                                             
def now_name(): return time.strftime("%m%d_%H%M%S", time.gmtime(time.time() + 8*60*60))
now_n = now_name()

test_path = 'data/train.csv'
test_data = [[np.fromstring(entry[1], sep=' '), entry[0]]  for entry in pd.read_csv(test_path).values] 
x_test, x_test_id = np.array(test_data).T 
x_test = np.concatenate(x_test).reshape(-1, 48, 48, 1).astype('float32') / 255 


for file in glob('./*.h5'):
    print(file, '......')
    model = load_model(file)
    res = model.predict_classes(x_test)
    with open('./'+file+'ppprediction_%s.csv'%now_n, 'w') as f:
            f.write('id,label\n')
            for n, prob in enumerate(res):
                f.write(str(n) + ',' + str(prob) + '\n')