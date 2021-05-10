# Used for testing the model

import os

GPUS='1'
os.system('export CUDA_VISIBLE_DEVICES='+GPUS)

BASE_ROOT='/content/Image_Text_Retrieval/deep_cmpl_model'
BASE_ROOT_2='drive/Shareddrives/Image-Text-Retrieval/tempckpt'
IMAGE_DIR='/content'
ANNO_DIR=BASE_ROOT+'/data/processed_data'
# CKPT_DIR=BASE_ROOT+'/data/model_data'
# LOG_DIR=BASE_ROOT+'/data/logs'
CKPT_DIR=BASE_ROOT_2+'/data/model_data'
LOG_DIR=BASE_ROOT_2+'/data/logs'
IMAGE_MODEL='efficient_net'
lr='0.0002'
batch_size='16'
lr_decay_ratio='0.9'
epoches_decay='80_150_200'


string = 'python3 {BASE_ROOT}/code/test.py --bidirectional --model_path {CKPT_DIR}/lr-{lr}-decay-{lr_decay_ratio}-batch-{batch_size} --image_model {IMAGE_MODEL} --log_dir {LOG_DIR}/lr-{lr}-decay-{lr_decay_ratio}-batch-{batch_size} --image_dir {IMAGE_DIR} --anno_dir {ANNO_DIR} --gpus {GPUS} --epoch_ema 0'.format(BASE_ROOT=BASE_ROOT, IMAGE_DIR=IMAGE_DIR, IMAGE_MODEL=IMAGE_MODEL, ANNO_DIR=ANNO_DIR, CKPT_DIR=CKPT_DIR, LOG_DIR=LOG_DIR, lr=lr, batch_size=batch_size, lr_decay_ratio=lr_decay_ratio, GPUS=GPUS)

os.system(string)

