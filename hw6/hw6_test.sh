wget "https://www.dropbox.com/s/cmai3gsqwbe45az/model_best_20190509_094032__20190509_093745.h5?dl=1"
wget                       "https://www.dropbox.com/s/p3ued1x4195g22c/saved_20190509_093745.model?dl=1"
wget "https://www.dropbox.com/s/j6ipdvh19ahe02g/model_best_20190509_094139__20190509_093754.h5?dl=1"
wget                       "https://www.dropbox.com/s/wxmpyag6rrn34ez/saved_20190509_093754.model?dl=1"
wget "https://www.dropbox.com/s/yz7l60l71z6rnwo/model_best_20190509_095448__20190509_095147.h5?dl=1"
wget                       "https://www.dropbox.com/s/n4nzychjeua3v3e/saved_20190509_095147.model?dl=1"
wget "https://www.dropbox.com/s/rd41i10vkvr8vit/model_best_20190509_095828__20190509_095536.h5?dl=1"
wget                       "https://www.dropbox.com/s/8s80f9hu9v9du2c/saved_20190509_095536.model?dl=1"

for i in ./*?=1; do mv $i ${i%%\?*}; done

python3 repro.py $1 $2 $3
