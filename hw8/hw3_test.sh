date
mkdir res

#mkdir weights
#cd weights
#curl -O https://github.com/jeffeuxMartin/ML-Files/releases/download/0.1.0/model_best_0408_010003.h5
#curl -O https://github.com/jeffeuxMartin/ML-Files/releases/download/0.1.0/model_best_0408_045244.h5
#curl -O https://github.com/jeffeuxMartin/ML-Files/releases/download/0.1.0/model_best_0409_044115.h5
#curl -O https://github.com/jeffeuxMartin/ML-Files/releases/download/0.1.0/model_best_0410_033221.h5
#curl -O https://github.com/jeffeuxMartin/ML-Files/releases/download/0.1.0/model_best_0410_034513.h5
#cd ..

python3 testing.py --testage "$1" --models weights --results res
python3 ensembler.py --results res --outputs "$2"
date