#!/bin/bash

fileid=1-3KRckCthLD8kOAjY_pb539GOdXYURzu
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -O #-o ${filename}

fileid=1-0G81BWYcmJesLWwsSS6gwjZvUz3NLfH
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -O #-o ${filename}

fileid=1r0l-mXbrWHEb6oUQ_0O4XwXP8p76v8Yk
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -O #-o ${filename}

mv word2vec* models

# bash news TD QS 
CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python3 src/Ensembled.py $1 $2 $3