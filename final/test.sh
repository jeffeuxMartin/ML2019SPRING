#!/bin/bash

if ! [ -e model/word2vec1500.model ]; then
    fileid=1-3KRckCthLD8kOAjY_pb539GOdXYURzu
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o model/word2vec1500.model
fi    

if ! [ -e model/word2vec1500.model.trainables.syn1neg.npy ]; then
    fileid=1-0G81BWYcmJesLWwsSS6gwjZvUz3NLfH
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o model/word2vec1500.model.trainables.syn1neg.npy
fi

if ! [ -e model/word2vec1500.model.wv.vectors.npy ]; then
    fileid=1r0l-mXbrWHEb6oUQ_0O4XwXP8p76v8Yk
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o model/word2vec1500.model.wv.vectors.npy
fi

if ! [ -e model/Q.json ]; then
    fileid=1-GAkBKMLLJWDhhdNxVwpbb5DmuPE_Y43
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o model/Q.json
fi

if [ -e cookie ]; then
	rm ./cookie
fi

# bash news TD QS 
CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python3 src/TFIDF_BOW.py $1 $2 $3 $4
CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python3 src/KWEmb.py $1 $2 $3 $4
CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python3 src/Ensembled.py $1 $2 $3 $4

if [ -e model/prob_tfidf.npy ]; then
	rm model/prob_tfidf.npy
fi
if [ -e model/prob_kwemb.npy ]; then
	rm model/prob_kwemb.npy
fi