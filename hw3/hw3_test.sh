echo date
mkdir res

mkdir weights
cd weights
curl -O https://github.com/jeffeuxMartin/ML2019SPRING/releases/download/0.0.0/model_best_0408_010003.h5
curl -O https://github.com/jeffeuxMartin/ML2019SPRING/releases/download/0.0.0/model_best_0408_045244.h5
curl -O https://github.com/jeffeuxMartin/ML2019SPRING/releases/download/0.0.0/model_best_0409_044115.h5
curl -O https://github.com/jeffeuxMartin/ML2019SPRING/releases/download/0.0.0/model_best_0410_033221.h5
curl -O https://github.com/jeffeuxMartin/ML2019SPRING/releases/download/0.0.0/model_best_0410_034513.h5
cd ..

python3 testing.py -te $1 -m weights -rs res
python3 ensembler.py -rs res -out $2
echo date