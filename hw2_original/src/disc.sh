if [ "$#" == 4 ]; then
    python3 train_ver1m1.py $1 $2 $3 $4;
elif [ "$#" == 2 ]; then
    python3 train_ver1m1.py 8e-5 500 $1 $2;
else
    echo lr iter weight res;
    python3 src/train_ver1m1.py 5e-5 10000 weights/newfi0.npy results/newfin3.csv
fi
