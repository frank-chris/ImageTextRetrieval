#!/bin/bash

# BASE_ROOT=/home/labyrinth7x/Codes/PersonSearch/Deep-Cross-Modal-Projection-Learning-for-Image-Text-Matching

BASE_ROOT=/mnt/c/Users/RAGHAV/Raghav_Goyal/Sem_6/Machine_Learning/project/Image_Text_Retrieval
IMAGE_ROOT=/mnt/c/Users/RAGHAV/Raghav_Goyal/Sem_6/Machine_Learning/project/Image_Text_Retrieval/dataset
JSON_ROOT=$BASE_ROOT/data/reid_raw.json
OUT_ROOT=$BASE_ROOT/data/processed_data

echo "Process Fashion dataset and save it as pickle form"

rm -rf $OUT_ROOT

python3 $BASE_ROOT/datasets/preprocess.py \
        --img_root=${IMAGE_ROOT} \
        --json_root=${JSON_ROOT} \
        --out_root=${OUT_ROOT} \
        --min_word_count 3 \
        --first
