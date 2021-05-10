GPUS=1
export CUDA_VISIBLE_DEVICES=$GPUS

<<<<<<< HEAD:scripts/shell_old/train.sh
BASE_ROOT=/mnt/c/Users/RAGHAV/Raghav_Goyal/Sem_6/Machine_Learning/project/Image_Text_Retrieval
IMAGE_DIR=/mnt/c/Users/RAGHAV/Raghav_Goyal/Sem_6/Machine_Learning/project/Image_Text_Retrieval/dataset
ANNO_DIR=$BASE_ROOT/data/processed_data
CKPT_DIR=$BASE_ROOT/data/model_data
LOG_DIR=$BASE_ROOT/data/logs
=======
BASE_ROOT=drive/Shareddrives/Image-Text-Retrieval/deepcca
IMAGE_DIR=/content
ANNO_DIR=$BASE_ROOT/data/processed_data
CKPT_DIR=$BASE_ROOT/data/model_data
LOG_DIR=$BASE_ROOT/data/logs
# PRETRAINED_PATH=$BASE_ROOT/pretrained_models/mobilenet.tar
#PRETRAINED_PATH=$BASE_ROOT/resnet50.pth
>>>>>>> 7d7a4ac943534fa53775eed173be3313ae889b49:dgpu/my_code/scripts/train.sh
IMAGE_MODEL=mobilenet_v1
lr=0.0002
num_epoches=300
batch_size=16
lr_decay_ratio=0.9
epoches_decay=80_150_200

<<<<<<< HEAD:scripts/shell_old/train.sh
python3 $BASE_ROOT/train.py \
=======
# python3 $BASE_ROOT/my_code/train.py \
python3 /content/train.py \
    --CMPC \
    --CMPM \
>>>>>>> 7d7a4ac943534fa53775eed173be3313ae889b49:dgpu/my_code/scripts/train.sh
    --bidirectional \
    --image_model $IMAGE_MODEL \
    --log_dir $LOG_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size \
    --checkpoint_dir $CKPT_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size \
    --image_dir $IMAGE_DIR \
    --anno_dir $ANNO_DIR \
    --batch_size $batch_size \
    --gpus $GPUS \
    --num_epoches $num_epoches \
    --lr $lr \
    --lr_decay_ratio $lr_decay_ratio \
    --epoches_decay ${epoches_decay} \
    --num_images 12305

