model_name=`date +"%y%m%d_%H%M%S"`
python3 first_try.py "model_${model_name}.h5"
python3 repr.py "model_${model_name}.h5" "pred_${model_name}.csv"