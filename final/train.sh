#!/bin/bash

if ! [ -e model/Q.json ]; then
    fileid=1-GAkBKMLLJWDhhdNxVwpbb5DmuPE_Y43
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o model/Q.json
fi

if [ -e cookie ]; then
	rm ./cookie
fi

# bash news TD QS 
CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python3 src/TrainEmb.py $1 $2 $3