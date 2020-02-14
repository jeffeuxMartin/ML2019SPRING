date
mkdir res

mkdir weights
cd weights

BASE="https://www.dropbox.com/s"
MID="model_best_04"
END="h5?dl=1"
curl "$BASE/k64olojglo1sd2v/${MID}08_010003.$END" -O -J -L
curl "$BASE/9nxkxfrvp691ot8/${MID}08_045244.$END" -O -J -L
curl "$BASE/o8jlicmnsm9usu6/${MID}09_044115.$END" -O -J -L
curl "$BASE/o01jcu0wuj6k5pt/${MID}10_033221.$END" -O -J -L
curl "$BASE/89ms5rxz314xg0n/${MID}10_034513.$END" -O -J -L
cd ..

python3 testing.py --testage "$1" \
                   --models weights --results res
python3 ensembler.py --results res --outputs "$2"
date