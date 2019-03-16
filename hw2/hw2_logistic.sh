'''
bash ./hw2_logistic.sh $1 $2 $3 $4 $5 $6	         output: your prediction
bash ./hw2_generative.sh $1 $2 $3 $4 $5 $6        output: your prediction
bash ./hw2_best.sh $1 $2 $3 $4 $5 $6                   output: your prediction
$1: raw data (train.csv)  $2: test data (test.csv)  
$3: provided train feature (X_train.csv)  $4: provided train label (Y_train.csv)
$5: provided test feature (X_test.csv)     $6: prediction.csv
'''

# python3 src/discriminative.py $3 $4 weights/model_log.npy
python3 src/testing.py -f weights/feat_scale.npy $5 $6 weights/model_log.npy