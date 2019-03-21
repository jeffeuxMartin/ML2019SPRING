#!/bin/bash
filenuml=1
filenumg=1
while [ -e results/pred_log${filenuml}.csv ]; do
	(( filenuml++ ));
done
while [ -e results/pred_log${filenumg}.csv ]; do
	(( filenumg++ ));
done
bash ./hw2_logistic_t.sh data/train.csv data/test.csv data/X_train data/Y_train data/X_test results/pred_log${filenuml}.csv
bash ./hw2_generative_t.sh data/train.csv data/test.csv data/X_train data/Y_train data/X_test results/pred_gen${filenumg}.csv