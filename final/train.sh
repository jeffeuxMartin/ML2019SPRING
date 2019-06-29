#!/bin/bash

fileid=1-3KRckCthLD8kOAjY_pb539GOdXYURzu
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o model/word2vec1500.model

fileid=1-0G81BWYcmJesLWwsSS6gwjZvUz3NLfH
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o model/word2vec1500.model.trainables.syn1neg.npy

fileid=1r0l-mXbrWHEb6oUQ_0O4XwXP8p76v8Yk
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o model/word2vec1500.model.wv.vectors.npy

fileid=1-GAkBKMLLJWDhhdNxVwpbb5DmuPE_Y43
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o model/Q.json

rm ./cookie
# bash news TD QS 
CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python3 src/Ensembled.py $1 $2 $3 $4
