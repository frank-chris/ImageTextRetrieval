#!/bin/bash

BASE_ROOT=/content/drive/Shareddrives/Image-Text-Retrieval/deepcca
IMAGE_ROOT=/content/dataset

JSON_ROOT=$BASE_ROOT/data/reid_raw.json
OUT_ROOT=$BASE_ROOT/data/processed_data

echo "Preprocessing dataset"

rm -rf $OUT_ROOT

python3 $BASE_ROOT/my_code/datasets/preprocess.py \
        --img_root=${IMAGE_ROOT} \
        --json_root=${JSON_ROOT} \
        --out_root=${OUT_ROOT} \
        --min_word_count 3 \
        --first
