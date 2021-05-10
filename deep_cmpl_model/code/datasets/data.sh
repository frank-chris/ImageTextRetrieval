#!/bin/bash

BASE_ROOT=Image_Text_Retrieval/deep_cmpl_model
IMAGE_ROOT=/content/dataset

JSON_ROOT=$BASE_ROOT/data/reid_raw.json
OUT_ROOT=$BASE_ROOT/data/processed_data

echo "Preprocessing dataset"

rm -rf $OUT_ROOT

python3 $BASE_ROOT/code/datasets/preprocess.py \
        --img_root=${IMAGE_ROOT} \
        --json_root=${JSON_ROOT} \
        --out_root=${OUT_ROOT} \
        --min_word_count 3 \
        --first
