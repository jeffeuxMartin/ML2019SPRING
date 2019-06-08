date
mkdir res

mkdir weights
cd weights
curl https://www.dropbox.com/s/k64olojglo1sd2v/model_best_0408_010003.h5?dl=1 -O -J -L
curl https://www.dropbox.com/s/9nxkxfrvp691ot8/model_best_0408_045244.h5?dl=1 -O -J -L
curl https://www.dropbox.com/s/o8jlicmnsm9usu6/model_best_0409_044115.h5?dl=1 -O -J -L
curl https://www.dropbox.com/s/o01jcu0wuj6k5pt/model_best_0410_033221.h5?dl=1 -O -J -L
curl https://www.dropbox.com/s/89ms5rxz314xg0n/model_best_0410_034513.h5?dl=1 -O -J -L
cd ..

python3 testing.py --testage "$1" --models weights --results res
python3 ensembler.py --results res --outputs "$2"
date